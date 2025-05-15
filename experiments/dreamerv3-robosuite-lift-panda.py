# main_robosuite.py
import gymnasium as gym # For wrappers
import torch
import argparse
import os
from tqdm import tqdm

# Robosuite imports
import robosuite as suite
from robosuite.wrappers import GymWrapper

# Project-specific imports
from dreamer import Dreamer
from utils import loadConfig, seedEverything, plotMetrics, saveLossesToCSV, ensureParentFolders
from envs import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper # Using getEnvProperties from envs.py

# Determine the device to use (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main(configFile):
    """
    Main function to run the Dreamer agent on a Robosuite environment.

    Args:
        configFile (str): Path to the YAML configuration file.
    """
    # --- 1. Load Configuration & Seed ---
    config = loadConfig(configFile)
    seedEverything(config.seed)
    print(f"Loaded configuration from: {configFile}")
    print(f"Using seed: {config.seed}")

    # --- 2. Define Run Name and File Paths ---
    # Create a more descriptive run name for Robosuite environments
    runName = f"{config.environmentName}_{config.robotName}_{config.runName}"
    
    # Define paths for checkpoints, metrics, plots, and videos
    checkpointFolder = os.path.join(config.folderNames.checkpointsFolder, runName)
    metricsFilename = os.path.join(config.folderNames.metricsFolder, f"{runName}.csv")
    plotFilename = os.path.join(config.folderNames.plotsFolder, f"{runName}.html")
    checkpointFilenameBase = os.path.join(checkpointFolder, runName) # Checkpoints will be saved in a subfolder per run
    videoFilenameBase = os.path.join(config.folderNames.videosFolder, runName)

    # Ensure parent directories for all output files exist
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    print(f"Run name: {runName}")
    print(f"Metrics will be saved to: {metricsFilename}")
    print(f"Plots will be saved to: {plotFilename}")
    print(f"Checkpoints will be saved to: {checkpointFolder}")
    print(f"Videos will be saved to: {config.folderNames.videosFolder}")

    # --- 3. Instantiate Robosuite Environment (Training) ---
    print("Initializing Robosuite training environment...")
    robosuite_env_train = suite.make(
        env_name=config.environmentName,
        robots=config.robotName,
        controller_configs=suite.load_composite_controller_config(controller=config.controllerName),
        has_renderer=False,  # No on-screen rendering for training
        has_offscreen_renderer=config.useCameraObs, # True if using camera observations for the encoder
        use_camera_obs=config.useCameraObs,
        camera_names=config.cameraName,
        camera_heights=config.cameraHeight, # Robosuite will render at this resolution
        camera_widths=config.cameraWidth,
        reward_shaping=config.rewardShaping,
        control_freq=config.controlFreq,
        horizon=config.horizon,
        # If only camera observation is used by the agent, specify keys for GymWrapper
        # This ensures GymWrapper returns the image array directly, not a dict.
        # The key for image observations in robosuite is typically f"{camera_name}_image".
    )
    
    # Wrap with GymWrapper to get an OpenAI Gym-compatible interface
    # If useCameraObs is true, keys should be set to extract only the image.
    train_env_keys = [f"{config.cameraName}_image"] if config.useCameraObs else None
    env = GymWrapper(robosuite_env_train, keys=train_env_keys)

    # Apply observation wrappers
    if config.useCameraObs:
        # ResizeObservation ensures the image is of the target size for the encoder.
        # Applied after GymWrapper, which provides the Gym-like interface.
        env = gym.wrappers.ResizeObservation(env, (config.cameraHeight, config.cameraWidth))
        # GymPixelsProcessingWrapper transposes (H,W,C) -> (C,H,W) and normalizes pixel values.
        env = GymPixelsProcessingWrapper(env)
    # CleanGymWrapper simplifies the step and reset API.
    env = CleanGymWrapper(env)
    print("Robosuite training environment initialized.")

    # --- 4. Instantiate Robosuite Environment (Evaluation) ---
    print("Initializing Robosuite evaluation environment...")
    robosuite_env_eval = suite.make(
        env_name=config.environmentName,
        robots=config.robotName,
        controller_configs=suite.load_controller_config(default_controller=config.controllerName),
        has_renderer=config.evaluationRender,  # Enable for video saving if needed
        has_offscreen_renderer=config.useCameraObs or config.evaluationRender, # Offscreen needed for rgb_array for video
        use_camera_obs=config.useCameraObs,
        camera_names=config.cameraName,
        camera_heights=config.cameraHeight,
        camera_widths=config.cameraWidth,
        reward_shaping=config.rewardShaping, # Usually good to keep consistent with training
        control_freq=config.controlFreq,
        horizon=config.horizon,
        # render_camera=config.cameraName # Or a different camera like "frontview" for videos
    )
    eval_env_keys = [f"{config.cameraName}_image"] if config.useCameraObs else None
    envEvaluation = GymWrapper(robosuite_env_eval, keys=eval_env_keys)

    if config.useCameraObs:
        envEvaluation = gym.wrappers.ResizeObservation(envEvaluation, (config.cameraHeight, config.cameraWidth))
        envEvaluation = GymPixelsProcessingWrapper(envEvaluation)
    envEvaluation = CleanGymWrapper(envEvaluation)
    print("Robosuite evaluation environment initialized.")

    # --- 5. Get Environment Properties ---
    # This function should work with the wrapped environment
    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(f"Environment Properties: Observation Shape {observationShape}, Action Size {actionSize}, Action Low {actionLow}, Action High {actionHigh}")
    if not config.useCameraObs:
        print("Warning: 'useCameraObs' is False. The current Dreamer implementation primarily expects image observations for its Encoder. Ensure your network setup is appropriate for state-based observations if this is intended.")

    # --- 6. Initialize Dreamer Agent ---
    print("Initializing Dreamer agent...")
    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config.dreamer)
    
    # --- 7. Load Checkpoint if Resuming ---
    if config.resume:
        checkpointToLoadPath = os.path.join(checkpointFolder, f"{runName}_{config.checkpointToLoad}.pth")
        try:
            dreamer.loadCheckpoint(checkpointToLoadPath)
            print(f"Resumed training from checkpoint: {checkpointToLoadPath}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint not found at {checkpointToLoadPath}. Starting from scratch.")
    else:
        print("Starting training from scratch.")

    # --- 8. Collect Initial Experiences ---
    print(f"Collecting {config.episodesBeforeStart} initial episodes before starting training...")
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)
    print(f"Buffer size after initial collection: {len(dreamer.buffer)}")

    # --- 9. Main Training Loop ---
    iterationsNum = config.gradientSteps // config.replayRatio
    print(f"Starting training for {config.gradientSteps} gradient steps, with {iterationsNum} outer iterations.")
    
    # Progress bar for the outer loop
    for iter_idx in tqdm(range(iterationsNum), desc="Training Progress"):
        # Inner loop for gradient updates
        for _ in range(config.replayRatio):
            if len(dreamer.buffer) < dreamer.config.batchSize * dreamer.config.batchLength: # Ensure enough data for a full sequence batch
                print(f"Buffer has {len(dreamer.buffer)} samples, less than required {dreamer.config.batchSize * dreamer.config.batchLength}. Collecting more...")
                dreamer.environmentInteraction(env, 1, seed=(config.seed + dreamer.totalEpisodes if config.seed else None)) # Collect one more episode
                if len(dreamer.buffer) < dreamer.config.batchSize * dreamer.config.batchLength:
                    continue # Skip training step if still not enough data

            sampledData = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics = dreamer.worldModelTraining(sampledData)
            behaviorMetrics = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1

            # Checkpoint saving and evaluation
            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps // 1000:.0f}k"
                currentCheckpointPath = f"{checkpointFilenameBase}_{suffix}.pth"
                dreamer.saveCheckpoint(currentCheckpointPath)
                
                # Perform evaluation
                video_path_suffix = f"{runName}_{suffix}"
                evaluationScore = dreamer.environmentInteraction(
                    envEvaluation, 
                    config.numEvaluationEpisodes, 
                    seed=(config.seed + dreamer.totalEpisodes if config.seed else None), # Use a different seed for eval if desired
                    evaluation=True, 
                    saveVideo=True, 
                    filename=os.path.join(config.folderNames.videosFolder, video_path_suffix) # Pass full path
                )
                print(f"Iter {iter_idx+1}/{iterationsNum} | Grad Steps {dreamer.totalGradientSteps} | Saved Checkpoint: {currentCheckpointPath} | Eval Score: {evaluationScore if evaluationScore is not None else 'N/A':>8.2f}")

        # Collect more environment interactions
        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=(config.seed + dreamer.totalEpisodes if config.seed else None))
        
        # Save metrics
        if config.saveMetrics:
            metricsBase = {
                "envSteps": dreamer.totalEnvSteps, 
                "gradientSteps": dreamer.totalGradientSteps, 
                "totalReward": mostRecentScore if mostRecentScore is not None else float('nan')
            }
            # Merge all metrics dictionaries
            all_metrics = {**metricsBase, **worldModelMetrics, **behaviorMetrics}
            saveLossesToCSV(metricsFilename, all_metrics)
            
            # Plot metrics (can be slow, consider doing it less frequently if performance is an issue)
            if dreamer.totalGradientSteps % (config.checkpointInterval * 5) == 0: # Plot less frequently
                 plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName} {config.robotName} - {config.runName}")

    print("Training finished.")
    # Final plot
    if config.saveMetrics:
        plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName} {config.robotName} - {config.runName} (Final)")
    env.close()
    envEvaluation.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dreamer agent on Robosuite environments.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="robosuite-lift-panda.yml",  # Default to the new Robosuite config
        help="Path to the configuration file (e.g., robosuite_lift_panda.yml)"
    )
    args = parser.parse_args()
    main(args.config)
