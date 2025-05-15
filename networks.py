import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)() # TanH or something

        # Recurrent model works with three inputs in total -> latent state, recurrent state and the action
        # It outputs a new recurrent state
        # Rs_1 = RecurrentModel(Rs_0, Ls_0, a_0)
        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize) # input size, hidden size

    def forward(self, recurrentState, latentState, action):
        return self.recurrent(self.activation(self.linear(torch.cat((latentState, action), -1))), recurrentState) # input = latent + action, hidden = recurrent state


class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1) # Convert logits to probabilities
        uniform = torch.ones_like(probabilities)/self.latentClasses # Create a uniform probability distribution for each discrete latent variable

        # This line performs a uniform mixing of probabilities. It's a regularization technique. 
        # By mixing the network's predicted probabilities with a small amount of a uniform distribution, we ensure:
        # No probability ever becomes exactly zero or one. This can help prevent the model from becoming overly confident and can improve numerical stability.
        # It encourages a small amount of exploration in the latent space, ensuring that all latent classes have at least a tiny chance of being selected. 
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform

        logits = probs_to_logits(finalProbabilities)
        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()

        return sample.view(-1, self.latentSize), logits
    

class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
 
        # The input is the full state, the output is the mean and std (given out as log std)
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)

        # Output is a normal distribution of the reward function.
        # Why? This allows the agent to sample a reward from the distribution, and also calculate the log probability to compute the loss
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config

        # Output is 1 dimensional - yes or no
        # Creates an MLP that outputs a single value (a logit)
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        # The output of the network is a logit.
        # It returns a Bernoulli distribution parameterized by these logits.
        # A Bernoulli distribution models a binary outcome (e.g., continue or terminate).
        # The distribution is expressed as (p for x = 1 | 1 - p for x = 0; where p is a probability)
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderConv(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels,            self.config.depth*1, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth*8*(height // (self.config.stride ** 4))*(width // (self.config.stride ** 4)), outputSize), 
            activation)

    def forward(self, x):
        return self.convolutionalNet(x).view(-1, self.outputSize)


class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, self.config.kernelSize,     self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, self.config.kernelSize + 1, self.config.stride),    activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels,       self.config.kernelSize + 1, self.config.stride))

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()

        # The network will output 2 values for each action dimension: mean and log_std
        # So, the actual output size of the MLP is actionSize * 2
        actionSize *= 2

        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)

        # Store action scaling and bias as buffers. These are not learned parameters
        # but are used to transform the network's output to the environment's action range.
        # actionScale = (actionHigh - actionLow) / 2.0
        # actionBias = (actionHigh + actionLow) / 2.0
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))

    def forward(self, x, training=False):
        # Define min and max for log_std to prevent std from becoming too small or too large
        logStdMin, logStdMax = -5, 2 

        mean, logStd = self.network(x).chunk(2, dim=-1)

        # Constrain logStd to be within [logStdMin, logStdMax]
        # 1. Apply tanh to logStd: output is in (-1, 1)
        # 2. Scale and shift: (tanh(logStd) + 1) / 2 maps it to (0, 1)
        # 3. Scale and shift again to map to [logStdMin, logStdMax]
        logStd = logStdMin + (logStdMax - logStdMin)*(torch.tanh(logStd) + 1)/2 # (-1, 1) to (min, max)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()

        # Apply tanh squashing to the sampled action. Output is in [-1, 1].
        # This helps in learning for bounded action spaces.
        sampleTanh = torch.tanh(sample)

        action = sampleTanh*self.actionScale + self.actionBias

        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config

        # The sequentialModel1D is configured to output 2 values (mean and log_std for the value distribution)
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        # The network's output is split into two equal parts along the last dimension.
        mean, logStd = self.network(x).chunk(2, dim=-1)
        # It returns a Normal (Gaussian) distribution representing the predicted value.
        # The mean is taken directly.
        # The standard deviation is derived by taking the exponent of logStd (to ensure positivity).
        # .squeeze(-1) removes the last dimension if it's 1, making the parameters suitable for Normal.
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
