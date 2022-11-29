from collections import deque

import random
import gym
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

E_START = 1
E_END = 0.1
E_STEPS_TO_END = 100000
MAX_STEPS = 50000000

BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4

MIN_MEM_SIZE = 2_000
MAX_MEMORY_SIZE = 10_000

UPDATE_FREQ = 4
NUM_FRAMES_STACK = 4
MAX_RANDOM_ACTIONS_RESET = 30
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


class RandomActionsOnReset(gym.Wrapper):
    def __init__(self, env, max_random_actions):
        super(RandomActionsOnReset, self).__init__(env)
        self.max_random_actions = max_random_actions

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(np.random.randint(1, self.max_random_actions + 1)):
            obs, _, _, _, info = self.env.step(self.env.action_space.sample())
        return obs, info


class ExperienceBuffer(object):
    def __init__(self, capacity, num_frames_stack):
        self.memory = deque(maxlen=capacity)

        self.max_capacity = capacity + 3
        self.num_frames_stack = num_frames_stack
        self.filled = False
        self.counter = 0
        self.new_episode = True

    def push(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def sample(self, num_samples, device):
        sample = random.sample(self.memory, k=num_samples)
        states = torch.stack([torch.Tensor(np.asarray(s[0])).to(device) for s in sample])
        actions = torch.tensor([s[1] for s in sample]).to(device)
        rewards = torch.tensor([s[2] for s in sample]).to(device)
        next_states = torch.stack([torch.Tensor(np.asarray(s[3])).to(device) for s in sample])
        dones = torch.tensor([s[4] for s in sample]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


def calc_epsilon(e_start, e_end, e_steps_to_anneal, steps_done):
    proportion_done = min((steps_done + 1) / (e_steps_to_anneal + 1), 1)
    return (proportion_done * (e_end - e_start)) + e_start


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init()

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=FRAMESKIP, repeat_action_probability=0)
    env = RandomActionsOnReset(env, MAX_RANDOM_ACTIONS_RESET)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, NUM_FRAMES_STACK)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    state, info = env.reset()

    policy_net = ConvNetwork(NUM_FRAMES_STACK, env.action_space.n).to(device)
    target_net = ConvNetwork(NUM_FRAMES_STACK, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ExperienceBuffer(MAX_MEMORY_SIZE, NUM_FRAMES_STACK)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    steps_done = 0
    while steps_done < MAX_STEPS:
        epsilon = calc_epsilon(E_START, E_END, E_STEPS_TO_END, steps_done)
        action = env.action_space.sample()
        if random.random() > epsilon:
            s = torch.Tensor(np.asarray(state)).to(device).unsqueeze(0)
            action = torch.argmax(policy_net(s), dim=1).cpu().numpy()[0]

        new_state, reward, done, trunc, info = env.step(action)

        memory.push(state, action, reward, new_state, done)

        state = new_state

        if "episode" in info.keys():
            wandb.log({"charts/episodic_return": info["episode"]["r"]})
            wandb.log({"charts/episodic_length": info["episode"]["l"]})
            wandb.log({"charts/epsilon": epsilon})
            wandb.log({"memory length": len(memory)})
            wandb.log({"steps per second": steps_done / (time.time() - start_time)})
            wandb.log({"steps_done": steps_done})

        if done:
            state, info = env.reset()

        if len(memory) >= MIN_MEM_SIZE and steps_done % UPDATE_FREQ == 0:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE, device)

            state_action_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_states_values = target_net(next_states).max(dim=1)[0]
                target_state_action_values = rewards + ((next_states_values * GAMMA) * (1 - dones.int()))

            loss = nn.MSELoss()(target_state_action_values, state_action_values)

            if steps_done % (100 * UPDATE_FREQ) == 0:
                wandb.log({"td_loss": loss})
                wandb.log({"q_values": state_action_values.mean().item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if steps_done % TARGET_NET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1
