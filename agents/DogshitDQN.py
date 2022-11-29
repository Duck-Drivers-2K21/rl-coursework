import random
from collections import deque
import time

import torch
import wandb
wandb.init(project="oHMYGOD")
import numpy as np
import gym
import torch.nn as nn
from torch import optim

LEARNING_RATE = 1e-4
TARGET_NET_UPDATE_FREQ = 1_000


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


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


class RandomActionsOnReset(gym.Wrapper):
    def __init__(self, env, max_random_actions):
        super(RandomActionsOnReset, self).__init__(env)
        self.max_random_actions = max_random_actions

    def reset(self):
        obs = self.env.reset()
        for _ in range(np.random.randint(1, self.max_random_actions + 1)):
            obs, _, _, info = self.env.step(self.env.action_space.sample())
        return obs


def run():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4, repeat_action_probability=0)
    env = RandomActionsOnReset(env, 30)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    start_time = time.time()

    min_buffer_size_to_learn = 1e3
    train_freq = 5
    batch_size = int(1e2)
    gamma = 0.9
    use_cuda = True

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    print(f"Using {device}.")

    # rb = ReplayBuffer(int(5e5), obs_size)
    rb = ExperienceBuffer(int(5e5), 4)

    # Epsilon params
    max_epsilon = 0.95
    min_epsilon = 0.05
    epsilon_decrease_ticks = 1_000_000
    d_epsilon = (max_epsilon - min_epsilon) / epsilon_decrease_ticks

    count = 0
    n_actions = env.action_space.n

    q_network = QNetwork(n_actions)
    target_network = QNetwork(n_actions)
    policy_network = QNetwork(n_actions)
    target_network.load_state_dict(policy_network.state_dict())
    optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    state = env.reset()
    episode = 0
    while True:
        # Decrease epsilon each tick, until we reach min epsilon
        epsilon = max(min_epsilon,
                      max_epsilon - (d_epsilon * count))

        # Exploit or explore?
        if random.random() < epsilon:
            # Randomly Explore
            action = env.action_space.sample()
        else:
            s = torch.Tensor(np.asarray(state)).to(device).unsqueeze(0)
            action = torch.argmax(policy_network(s), dim=1).cpu().numpy()[0]

        next_state, reward, terminal, info = env.step(action)

        if "episode" in info.keys():
            wandb.log({"episodic_return": info["episode"]["r"]})
            wandb.log({"episodic_length": info["episode"]["l"]})
            wandb.log({"epsilon": epsilon})
            wandb.log({"memory length": len(rb)})
            wandb.log({"steps per second": count / (time.time() - start_time)})
            wandb.log({"steps_done": count})

        if terminal:
            # if (episode % 100) == 0:
            #     wandb.log({"video": wandb.Video(f'videos/rl-video-episode-{episode}.mp4', fps=30, format="mp4")})
            episode += 1
            state = env.reset()

        # Save this new info to the Replay buffer
        rb.push(state, action, reward, next_state, terminal)

        if count >= min_buffer_size_to_learn and count % train_freq == 0:
            # get sample data from replay buffer
            states, batch_action, batch_reward, batch_ns, batch_done = rb.sample(batch_size, device)

            state_action_values = policy_network(states).gather(1, batch_action.unsqueeze(1)).squeeze()

            # Disable torch gradients
            with torch.no_grad():
                # Tensor of state-action values
                target_max = target_network(states).max(dim=1).values

                # 0 if state is terminal, 1 if not
                done_mask = 1 - batch_done.int()
                td_target = batch_reward + (gamma * target_max * done_mask)

            loss = nn.MSELoss()(td_target, state_action_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if count % TARGET_NET_UPDATE_FREQ == 0:
            target_network.load_state_dict(policy_network.state_dict())

        # Setup for next iterations
        state = next_state
        count += 1


if __name__ == "__main__":
    run()
