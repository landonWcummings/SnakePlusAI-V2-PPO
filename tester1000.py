from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from snake_env import SnakeEnv
import time
import os
import glob
import matplotlib.pyplot as plt

def get_latest_checkpoint(checkpoint_dir):
    """Returns the path to the latest checkpoint file in the directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'snake_ppo_*.zip'))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getmtime)  # Get the most recent file

def test_agent(num_episodes=1000):
    # Use headless environment
    env = SnakeEnv(gridsize=6, headless=True)
    path = get_latest_checkpoint(r"C:\Users\lndnc\OneDrive\Desktop\checkpoints")
    if path is None:
        print("No checkpoint found.")
        return
    
    model = PPO.load(path)

    scores = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
        scores.append(env.score)
    
    env.close()

    # Plot the score distribution
    plt.hist(scores, bins='auto', edgecolor='black')
    plt.title('Score Distribution Over {} Episodes'.format(num_episodes))
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test_agent()  # Run 1000 headless test episodes and plot score distribution
