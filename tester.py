from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from snake_env import SnakeEnv
import time
import os
import glob

def get_latest_checkpoint(checkpoint_dir):
    """Returns the path to the latest checkpoint file in the directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'snake_ppo_*.zip'))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getmtime)  # Get the most recent file

def test_agent():
    env = SnakeEnv(gridsize=6, headless=False)
    path = get_latest_checkpoint(r"C:\Users\lndnc\OneDrive\Desktop\checkpoints")
    if path is None:
        print("No checkpoint found.")
        return
    
    model = PPO.load(path)
    obs, _ = env.reset()  # Unpack observation and info
    done = False

    while not done:
        action, _ = model.predict(obs)  # Pass only the observation
        obs, reward, done, truncated, info = env.step(action)
        print(obs)
        env.render()
        time.sleep(0.03)  # Adjust speed as needed

    env.close()


if __name__ == '__main__':
    #train_agent()  # Train the agent
    test_agent()   # Test the trained agent
