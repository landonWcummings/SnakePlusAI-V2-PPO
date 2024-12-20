from stable_baselines3 import PPO
import onnx


model = PPO.load(r'C:\Users\lndnc\OneDrive\Desktop\AI test\snakePPO\checkpoints\snake_ppo_196200000_steps.zip')

import torch
class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super(PolicyWrapper, self).__init__()
        self.policy = policy

    def forward(self, obs):
        # Forward pass through the policy to get the deterministic action
        distribution = self.policy.get_distribution(obs)
        action = distribution.mode()  # Get the deterministic action
        return action


dummy_input = torch.zeros(1,25)

torch.onnx.export(
    PolicyWrapper(model.policy),
    dummy_input,
    'ppo_policy5.onnx',
    input_names=['obs'],
    output_names=['action'],
    opset_version=14
)

model = onnx.load('ppo_policy.onnx')
print("Model Inputs:", [input.name for input in model.graph.input])
print("Model Outputs:", [output.name for output in model.graph.output])
onnx.checker.check_model(model)
