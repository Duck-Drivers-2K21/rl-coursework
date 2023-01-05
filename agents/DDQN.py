from collections import deque

import random
import gymnasium
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from wrappers import RandomActionsOnReset


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


def run():
    wandb.init(project="oHMYGOD")
    env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4, repeat_action_probability=0)
    env = RandomActionsOnReset(env, 30)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)
    env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
    env = gymnasium.wrappers.GrayScaleObservation(env)
    env = gymnasium.wrappers.FrameStack(env, 4)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    state, info = env.reset()

    start_time = time.time()

    min_buffer_size_to_learn = 80_000
    train_freq = 4
    batch_size = int(1e2)
    gamma = 0.9
    use_cuda = True

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using {device}.")

    rb = ExperienceBuffer(int(1_000_000), 4)

    # Epsilon params
    max_epsilon = 0.95
    min_epsilon = 0.05
    epsilon_decrease_ticks = 1_000_000
    d_epsilon = (max_epsilon - min_epsilon) / epsilon_decrease_ticks

    count = 0
    n_actions = env.action_space.n

    target_network_1 = QNetwork(n_actions).to(device)
    target_network_2 = QNetwork(n_actions).to(device)

    policy_network_1 = QNetwork(n_actions).to(device)
    policy_network_2 = QNetwork(n_actions).to(device)

    target_network_1.load_state_dict(policy_network_1.state_dict())
    target_network_2.load_state_dict(policy_network_2.state_dict())

    optimizer_1 = optim.Adam(policy_network_1.parameters(), lr=LEARNING_RATE)
    optimizer_2 = optim.Adam(policy_network_2.parameters(), lr=LEARNING_RATE)

    episode = 0
    while True:
        # Select which networks to use (DDQN)
        even_count = (count % 2 == 0)
        policy_network, other_policy_network = ((policy_network_1, policy_network_2) if even_count
                                                else (policy_network_2, policy_network_1))
        target_network = target_network_1 if even_count else target_network_2
        optimizer = optimizer_1 if even_count else optimizer_2

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

        next_state, reward, terminal, _, info = env.step(action)

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
            state = env.reset()[0]

        # Save this new info to the Replay buffer
        rb.push(state, action, reward, next_state, terminal)

        if count >= min_buffer_size_to_learn and count % train_freq == 0:
            # get sample data from replay buffer
            states, batch_action, batch_reward, batch_ns, batch_done = rb.sample(batch_size, device)

            # Get the State-action pair values from the Policy Network
            policy_state_action_values = policy_network(states).gather(1, batch_action.unsqueeze(1)).squeeze()

            # Disable torch gradients
            with torch.no_grad():
                best_actions = torch.argmax(other_policy_network(states), dim=1).unsqueeze(-1)

                # Get the state-action pair values from the Target Network
                target_max = target_network(states).gather(1, best_actions).squeeze(1)

                # 0 if state is terminal, 1 if not
                done_mask = 1 - batch_done.int()

                target_network_TD = batch_reward + (gamma * target_max * done_mask)

            # Calculate the Loss between the real stat
            loss = nn.MSELoss()(target_network_TD, policy_state_action_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if count % TARGET_NET_UPDATE_FREQ == 0:
            target_network_1.load_state_dict(policy_network_1.state_dict())
            target_network_2.load_state_dict(policy_network_2.state_dict())

        # Setup for next iterations
        state = next_state
        count += 1


if __name__ == "__main__":
    run()
