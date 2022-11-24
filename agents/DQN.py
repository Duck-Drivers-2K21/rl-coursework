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
E_END = 0.01
E_STEPS_TO_END = 1_100_000
MAX_STEPS = 10_000_000

BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4

MIN_MEM_SIZE = 80_000
MAX_MEMORY_SIZE = 700_000

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
            obs, _, _, _, info = self.env.step(0)
        return obs, info


class CroppedBorders(gym.Wrapper):
    def __init__(self, env):
        super(CroppedBorders, self).__init__(env)

    def _preprocess_frame(observation: np.ndarray) -> np.ndarray:
        # Remove scores and cut unecessary whitespace...
        assert observation.shape == (210, 160), "Sizes don't match..."
        ROWS = [194 + j for j in range(16)] + [i for i in range(34)]
        cropped_frame = np.delete(observation, ROWS, axis=0)
        # print(cropped_frame.shape)  # We've ended up with a 160x160 input...
        return cropped_frame


class ExperienceBuffer(object):
    def __init__(self, capacity):
        #self.memory = deque(maxlen=capacity)

        self.max_capacity = capacity + NUM_FRAMES_STACK

        self.frames = np.ndarray((self.max_capacity, 84, 84), dtype=np.single)
        self.actions = np.ndarray((self.max_capacity, 1), dtype=int)
        self.rewards = np.ndarray((self.max_capacity, 1), dtype=int)
        self.dones = np.ndarray((self.max_capacity, 1), dtype=int)

        self.filled = False
        self.counter = 0
        self.new_episode = True

    def push(self, state, action, reward, new_state, done):
        #self.memory.append((state, action, reward, new_state, done))

        def increment_counter():
            if self.counter + 1 == self.max_capacity:
                self.filled = True
            self.counter = (self.counter + 1) % self.max_capacity

        if self.new_episode:
            for frame_num in range(NUM_FRAMES_STACK):
                self.frames[self.counter] = np.asarray(state)[frame_num]
                self.actions[self.counter] = action
                self.rewards[self.counter] = reward
                self.dones[self.counter] = done
                increment_counter()

        self.frames[self.counter] = np.asarray(new_state)[-1]
        self.actions[self.counter] = action
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done
        increment_counter()

        self.new_episode = done

    def sample(self, num_samples, device):
        if self.filled:
            indices = (np.random.randint(NUM_FRAMES_STACK - 1, self.max_capacity, size=num_samples) + self.counter) % self.max_capacity
        else:
            indices = np.random.randint(0, self.counter, size=num_samples)

        next_states = np.ndarray((num_samples, NUM_FRAMES_STACK, 84, 84))
        states = np.ndarray((num_samples, NUM_FRAMES_STACK, 84, 84))
        actions = np.ndarray((num_samples,))
        rewards = np.ndarray((num_samples,))
        dones = np.ndarray((num_samples,))

        for sample_num, i in enumerate(indices):
            frames = np.ndarray((NUM_FRAMES_STACK + 1, 84, 84))
            for frame in range(NUM_FRAMES_STACK + 1):
                frames[frame] = self.frames[(i + frame - NUM_FRAMES_STACK + 1) % self.max_capacity]
            states[sample_num] = frames[:-1]
            next_states[sample_num] = frames[1:]
            actions[sample_num] = self.actions[i]
            rewards[sample_num] = self.rewards[i]
            dones[sample_num] = self.dones[i]

        return torch.as_tensor(states, dtype=torch.float, device=device), \
               torch.as_tensor(actions, dtype=torch.int64, device=device), \
               torch.as_tensor(rewards, dtype=torch.int64, device=device), \
               torch.as_tensor(next_states, dtype=torch.float, device=device), \
               torch.as_tensor(dones, dtype=torch.int64, device=device)

    def __len__(self):
        return self.max_capacity if self.filled else self.counter - 1


def calc_epsilon(e_start, e_end, e_steps_to_anneal, steps_done):
    proportion_done = min((steps_done + 1) / (e_steps_to_anneal + 1), 1)
    return (proportion_done * (e_end - e_start)) + e_start


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(config={
        "frameskip": FRAMESKIP,
        "framestack": NUM_FRAMES_STACK,
        "epsilon_start": E_START,
        "epsilon_end": E_END,
        "epsilon_steps_to_anneal": E_STEPS_TO_END,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "update_frequency": UPDATE_FREQ,
        "update_target_net_steps": TARGET_NET_UPDATE_FREQ,
        "memory_start_training_size": MIN_MEM_SIZE,
        "memory_max_size": MAX_MEMORY_SIZE,
        "max_random_actions_on_reset": MAX_RANDOM_ACTIONS_RESET,
        "seed": SEED
    })

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=FRAMESKIP, repeat_action_probability=0)
    env = RandomActionsOnReset(env, MAX_RANDOM_ACTIONS_RESET)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = CroppedBorders(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, NUM_FRAMES_STACK)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    state, info = env.reset()

    policy_net = ConvNetwork(NUM_FRAMES_STACK, env.action_space.n).to(device)
    target_net = ConvNetwork(NUM_FRAMES_STACK, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ExperienceBuffer(MAX_MEMORY_SIZE)
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
            wandb.log({"episodic_return": info["episode"]["r"]})
            wandb.log({"episodic_length": info["episode"]["l"]})
            wandb.log({"epsilon": epsilon})
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
                target_state_action_values = rewards + ((next_states_values * GAMMA) * (1 - dones))

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
