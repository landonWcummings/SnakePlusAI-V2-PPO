from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv
import numpy as np
import time
import os
from glob import glob
from stable_baselines3.common.vec_env import SubprocVecEnv


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.step_count = 0  # Count environment steps

    def _on_step(self) -> bool:
        self.step_count += 1

        # Check if an episode finished in the current environment
        if "episode" in self.locals["infos"][0]:
            episode_reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(episode_reward)
            # Log the average reward over the last 10 episodes
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.logger.record("average_reward_last_10_episodes", avg_reward)

        # Every 5 million steps, log the average reward across all episodes so far
        if self.step_count % 5_000_000 == 0 and len(self.episode_rewards) > 0:
            avg_reward_5m = np.mean(self.episode_rewards)
            self.logger.record("5M_step_avg_reward", avg_reward_5m)

        return True


def get_latest_checkpoint(checkpoint_dir):
    """Returns the path to the latest checkpoint file in the directory."""
    checkpoint_files = glob(os.path.join(checkpoint_dir, 'snake_ppo_*.zip'))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getmtime)  # Get the most recent file


def train_agent(resume=False):
    # Ensure checkpoint directory exists
    checkpoint_dir = r'c:\Users\lndnc\OneDrive\Desktop\checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Use DummyVecEnv to support multiple environments
    env = SubprocVecEnv([lambda: Monitor(SnakeEnv(gridsize=6, headless=True)) for _ in range(8)])
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path=checkpoint_dir,
        name_prefix='snake_ppo'
    )

    model = None
    if resume:
        # Attempt to load the latest checkpoint
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            model = PPO.load(latest_checkpoint, env=env, device="cuda")
        else:
            print("No checkpoint found. Starting training from scratch.")
    
    if model is None:
        # Initialize a new model if no checkpoint is found
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            n_steps=2048,  # Larger rollout buffer
            batch_size=1024,  # Matches the rollout buffer size
            tensorboard_log="./ppo_snake_tensorboard/",
            device="cuda"
        )
    
    model.learn(total_timesteps=6000000000, callback=[checkpoint_callback, TensorboardCallback()])
    model.save('snake_ppo')


def test_agent():
    env = SnakeEnv(gridsize=6, headless=False)
    model = PPO.load('snake_ppo')
    obs, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)  # Adjust speed as needed
    env.close()


if __name__ == '__main__':
    resume_training = False  # Set to True to resume training from the last checkpoint
    train_agent(resume=resume_training)
    # test_agent()
