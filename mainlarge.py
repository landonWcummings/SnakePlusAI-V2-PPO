from snake_env import SnakeEnv
import gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.envs import DummyVecEnv


# Custom MLP architecture
class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # Larger number of neurons
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Custom Policy to integrate with PPO
class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomPolicy, self).__init__(observation_space, features_dim)
        input_dim = gym.spaces.utils.flatdim(observation_space)
        self.mlp = CustomMLP(input_dim, features_dim)

    def forward(self, observations):
        return self.mlp(observations)

# Training function
def train_agent():
    # Create the Snake environment
    env = DummyVecEnv([lambda: SnakeEnv(gridsize=6, headless=True)])

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=r'C:\Users\lndnc\OneDrive\Desktop\AI test\snakePPO\checkpoints\\',
        name_prefix='snake_ppo'
    )

    # Define custom policy arguments
    policy_kwargs = dict(
        features_extractor_class=CustomPolicy,
        features_extractor_kwargs=dict(features_dim=256)
    )

    # Create PPO model with the custom policy
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        batch_size=128,
        n_steps=2048,  # Adjust for longer rollouts
        tensorboard_log="./ppo_snake_tensorboard/",
        device="cuda",
        policy_kwargs=policy_kwargs  # Pass custom policy
    )

    # Train the model
    model.learn(total_timesteps=200000000, callback=checkpoint_callback)
    model.save('snake_ppo')

# Run the training process
if __name__ == "__main__":
    train_agent()
