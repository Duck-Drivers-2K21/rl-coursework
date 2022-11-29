import random
import gymnasium
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

E_START = 1
E_END = 0.01
E_STEPS_TO_END = 1_100_000
MAX_STEPS = 5_000_000

BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4

MIN_MEM_SIZE = 80_000
MAX_MEMORY_SIZE = 1_000_000

UPDATE_FREQ = 4
NUM_FRAMES_STACK = 4
MAX_NOOPS = 30
FRAMESKIP = 4

TARGET_NET_UPDATE_FREQ = 1_000

SEED = 1


class ConvNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )

    def forward(self, x):
        return self.conv(x / 255.0)


class CroppedBorders(gymnasium.Wrapper):
    def __init__(self, env):
        super(CroppedBorders, self).__init__(env)

    def _preprocess_frame(observation: np.ndarray) -> np.ndarray:
        # Remove scores and cut unecessary whitespace...
        assert observation.shape == (210, 160), "Sizes don't match..."
        ROWS = [194 + j for j in range(16)] + [i for i in range(34)]
        cropped_frame = np.delete(observation, ROWS, axis=0)
        # print(cropped_frame.shape)  # We've ended up with a 160x160 input...
        return cropped_frame


env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=FRAMESKIP, repeat_action_probability=0)
env = CroppedBorders(env)
env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
env = gymnasium.wrappers.GrayScaleObservation(env)
env = gymnasium.wrappers.FrameStack(env, NUM_FRAMES_STACK)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = ConvNetwork(NUM_FRAMES_STACK, env.action_space.n).to(device)
policy_net.load_state_dict(torch.load("latest_model.nn"))

MAX_EPISODES = 1000

total_rewards = 0
episode_num = 0
for episode_num in range(MAX_EPISODES):
    print(episode_num, total_rewards / max(episode_num, 1))
    state, info = env.reset()

    done = False
    while not done:
        s = torch.Tensor(np.asarray(state)).to(device).unsqueeze(0)
        action = torch.argmax(policy_net(s), dim=1).cpu().numpy()[0]

        state, reward, done, trunc, info = env.step(action)
        total_rewards += reward

print(total_rewards)
print(total_rewards / MAX_EPISODES)