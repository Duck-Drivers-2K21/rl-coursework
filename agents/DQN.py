import random
import gymnasium
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from wrappers import CroppedBorders, NoopsOnReset


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


class ExperienceBuffer(object):
    def __init__(self, capacity):
        self.max_capacity = capacity + NUM_FRAMES_STACK

        self.frames = np.ndarray((self.max_capacity, 84, 84), dtype=np.uint8)
        self.actions = np.ndarray((self.max_capacity,), dtype=int)
        self.rewards = np.ndarray((self.max_capacity,), dtype=int)
        self.dones = np.ndarray((self.max_capacity,), dtype=int)

        self.valid_frame = np.ndarray((self.max_capacity,), dtype=bool)

        self.filled = False
        self.counter = 0
        self.new_episode = True

    def push(self, state, action, reward, new_state, done):
        def increment_counter():
            if self.counter + 1 == self.max_capacity:
                self.filled = True
            self.counter = (self.counter + 1) % self.max_capacity

        if self.new_episode:
            for frame_num in range(NUM_FRAMES_STACK):
                self.frames[self.counter] = np.asarray(state)[frame_num]
                self.valid_frame[self.counter] = False
                increment_counter()
            for frame_num in range(NUM_FRAMES_STACK):
                self.valid_frame[self.counter + frame_num] = False

        self.valid_frame[self.counter - 1] = True
        self.actions[self.counter - 1] = action
        self.rewards[self.counter - 1] = reward
        self.dones[self.counter - 1] = done

        self.frames[self.counter] = np.asarray(new_state)[-1]
        self.valid_frame[(self.counter + NUM_FRAMES_STACK - 1) % self.max_capacity] = False
        increment_counter()

        self.new_episode = done

    def sample(self, num_samples, device):
        if self.filled:
            indices = np.random.randint(self.max_capacity, size=num_samples)
        else:
            indices = np.random.randint(0, self.counter, size=num_samples)

        valid_samples = sum([self.valid_frame[i] for i in indices])
        next_states = np.ndarray((valid_samples, NUM_FRAMES_STACK, 84, 84), dtype=float)
        states = np.ndarray((valid_samples, NUM_FRAMES_STACK, 84, 84), dtype=float)
        actions = np.ndarray((valid_samples,), dtype=int)
        rewards = np.ndarray((valid_samples,), dtype=int)
        dones = np.ndarray((valid_samples,), dtype=int)

        invalid_frames_passed = 0
        for sample_num, i in enumerate(indices):
            sample_num = sample_num - invalid_frames_passed
            if self.valid_frame[i]:
                frames = np.ndarray((NUM_FRAMES_STACK + 1, 84, 84))
                for frame in range(NUM_FRAMES_STACK + 1):
                    frames[frame] = self.frames[(i + frame - NUM_FRAMES_STACK + 1) % self.max_capacity]
                states[sample_num] = frames[:-1]
                next_states[sample_num] = frames[1:]
                actions[sample_num] = self.actions[i]
                rewards[sample_num] = self.rewards[i]
                dones[sample_num] = self.dones[i]
            else:
                invalid_frames_passed += 1

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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
        "max_noops": MAX_NOOPS,
        "seed": SEED
    })

    env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=FRAMESKIP, repeat_action_probability=0)
    env = gymnasium.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 100 == 0)
    env = NoopsOnReset(env, MAX_NOOPS)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)
    env = CroppedBorders(env)
    env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
    env = gymnasium.wrappers.GrayScaleObservation(env)
    env = gymnasium.wrappers.FrameStack(env, NUM_FRAMES_STACK)
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
    episode = 0
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
            if (episode % 100) == 0:
                wandb.log({"video": wandb.Video(f'videos/rl-video-episode-{episode}.mp4', fps=30, format="mp4")})
            episode += 1
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
