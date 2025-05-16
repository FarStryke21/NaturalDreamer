import gymnasium as gym
import numpy as np

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs

class ImageExtractWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_key):
        super().__init__(env)
        self.image_key = image_key

        if not isinstance(self.env.observation_space, gym.spaces.Dict):
            raise ValueError(f"ImageExtractWrapper requires the wrapped environment to have a Dict observation space. Got: {type(self.env.observation_space)}")
        if self.image_key not in self.env.observation_space.spaces:
            raise ValueError(f"Image key '{self.image_key}' not found in the wrapped environment's observation space keys: {list(self.env.observation_space.spaces.keys())}")

        image_space = self.env.observation_space.spaces[self.image_key]
        if not isinstance(image_space, gym.spaces.Box):
            raise ValueError(f"The observation for key '{self.image_key}' is not a Box space. Got: {type(image_space)}")

        # The new observation space is just the image's Box space
        self.observation_space = image_space
        # print(f"ImageExtractWrapper: New observation_space is {self.observation_space}")


    def observation(self, observation_dict):
        if not isinstance(observation_dict, dict) or self.image_key not in observation_dict:
            # This might happen if the underlying environment's observation format changes unexpectedly
            raise ValueError(f"Expected a dictionary observation with key '{self.image_key}', but got: {observation_dict}")
        return observation_dict[self.image_key]