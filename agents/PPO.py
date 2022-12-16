import time
import torch
import wandb
import numpy as np
import gymnasium

import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from wrappers import CroppedBorders, NoopsOnReset


def make_env(seed, idx, max_noops, num_frames_stack, frameskip):
    def create_env():
        env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=frameskip, repeat_action_probability=0)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env = CroppedBorders(env)
        if idx == 0:
            env = gymnasium.wrappers.RecordVideo(env, "videos_ppo")
        env = NoopsOnReset(env, max_noops)
        env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
        env = gymnasium.wrappers.GrayScaleObservation(env)
        env = gymnasium.wrappers.FrameStack(env, num_frames_stack)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return create_env

# Network from cleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
class Network(nn.Module):
    def __init__(self, num_frame_stack, learning_rate):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            self.layer_init(nn.Conv2d(in_channels=num_frame_stack, out_channels=32, kernel_size=8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        # std of 0.01 ensure the output layers will have similar scalar values
        # making probability of taking each action to be similar
        self.actor = self.layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)
        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    # pytorch by default uses different layer initialisation methods
    # therefore we must specify to be inline with PPO
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.conv(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.conv(x / 255.0)
        logits = self.actor(hidden)  # unormalised action probs
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPOAgent:

    def __init__(self, envs, args, device):
        self.args = args
        self.network = Network(self.args['num_frames_stack'],
                               self.args['learning_rate']).to(device)

        self.states = torch.zeros((self.args['num_steps'], self.args['num_env']) + envs.single_observation_space.shape)\
            .to(device)
        self.actions = torch.zeros((self.args['num_steps'], self.args['num_env'])).to(device)
        self.logprobs = torch.zeros((self.args['num_steps'], self.args['num_env'])).to(device)
        self.rewards = torch.zeros((self.args['num_steps'], self.args['num_env'])).to(device)
        self.dones = torch.zeros((self.args['num_steps'], self.args['num_env'])).to(device)
        self.values = torch.zeros((self.args['num_steps'], self.args['num_env'])).to(device)

    def rollout(self, curr_states, curr_dones, step):
        self.states[step] = curr_states
        self.dones[step] = curr_dones

        with torch.no_grad():  # we dont need to cache gradient in rollout
            action, logprob, _, value = self.network.get_action_and_value(curr_states)
            self.values[step] = value.flatten()
        self.actions[step] = action
        self.logprobs[step] = logprob

        new_next_states, reward, new_dones, _, new_info = envs.step(action.cpu().numpy())
        self.rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_states_tensor = torch.tensor(new_next_states).to(device)
        next_done_tensor = torch.tensor(new_dones, dtype=float).to(device)
        return next_states_tensor, next_done_tensor, new_info

    # find the expected return - actual return
    def calc_advantages(self, num_steps, gamma, next_states):
        with torch.no_grad():
            next_value = self.network.get_value(next_states).reshape(1, -1)  # gets value of next_states
            returns = torch.zeros_like(self.rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1 - self.dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = self.rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - self.values
        return advantages, returns

    def generate_minibatches(self):
        batch_indices = np.arange(self.args['batch_size'])
        np.random.shuffle(batch_indices)
        batches = []
        for start in range(0, self.args['batch_size'], self.args['minibatch_size']):
            end = start + self.args['minibatch_size']
            minibatch_indicies = batch_indices[start:end]
            batches.append((start, end, minibatch_indicies))
        return batches

    def learn(self, envs, next_states):

        for epoch in range(args['num_epochs']):
            # Advantage is the difference in predicted reward and actual value
            # Returns is what we get from the values predicted by the network
            advantages, returns = self.calc_advantages(self.args['num_steps'], self.args['gamma'], next_states)

            advantages = advantages.flatten()
            returns = returns.flatten()

            # get stored state values
            logprobs = self.logprobs.flatten()
            states = self.states.reshape((-1,) + envs.single_observation_space.shape)
            values = self.values.flatten()
            actions = self.actions.flatten()

            batches = self.generate_minibatches()
            val_loss, policy_loss, entropy_loss = 0, 0, 0

            # Use minibatches to optimise network
            for start, end, minibatch_indicies in batches:
                # forward pass into network
                _, newlogprob, entropy, new_val = self.network.get_action_and_value(
                    states[minibatch_indicies], actions.long()[minibatch_indicies]
                    # pass minibatch actions so agent doesnt sample new ones
                )

                new_val = new_val.flatten()

                logratio = newlogprob - logprobs[minibatch_indicies]
                ratio = logratio.exp()

                with torch.no_grad():
                    # divergence - says how aggressive the update is
                    approx_kl = (-logratio).mean()

                # probability clipping
                weighted_probs = advantages[minibatch_indicies] * ratio
                weighted_clipped_probs = torch.clamp(ratio, 1 - self.args['clip_co'], 1 + self.args['clip_co']) \
                                         * advantages[minibatch_indicies]
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # MSE currently but could try something else
                # TODO: turn loss calculation into its own function then can test diff loss funcs.
                val_loss = ((new_val - returns[minibatch_indicies]) ** 2).mean()

                entropy_loss = entropy.mean()

                # minimise polilicy loss, value loss and maximise entropy loss
                # larger entropy is larger exploration
                loss = policy_loss - self.args['ent_co'] * entropy_loss + val_loss * self.args['vl_co']

                self.network.optimiser.zero_grad()
                loss.backward()
                self.network.optimiser.step()

        # returns debug variables for wandb
        return approx_kl, values, returns, val_loss, policy_loss, entropy_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    args = {
        "seed": 1,
        "learning_rate": 2.5e-4,
        'num_env': 1,
        'num_steps': 128,
        'total_timesteps': 25000000,  # set arbitrarily high
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'num_minibatch': 4,
        'num_epochs': 4,
        'clip_co': 0.2,
        'ent_co': 0.01,
        'vl_co': 0.5,
        'max_noops': 30,
        'num_frames_stack': 4,
        'frameskip': 4,
    }

    args['batch_size'] = args['num_env'] * args['num_steps']
    args['minibatch_size'] = int(args['num_steps'] // args['num_minibatch'])

    wandb.init(project="test pong again", entity="lc2232", config=args)

    envs = gymnasium.vector.SyncVectorEnv(
        [make_env(args['seed'], i, args['max_noops'], args['num_frames_stack'], args['frameskip']) for i in
         range(args['num_env'])])

    agent = PPOAgent(envs, args, device)

    # game start
    global_step = 0
    start_time = time.time()
    next_states = torch.tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args["num_env"]).to(device)
    num_updates = args['total_timesteps'] // args['batch_size']

    # training loop:

    # each update involves, interacting with env (for a num_steps), training policy and updating agent
    for update in range(1, num_updates + 1):

        # policy rollout (doing num steps in env)
        # each policy step occurs in each vector env so we inc global step by num envs
        for step in range(0, args['num_steps']):
            global_step += 1 * args['num_env']  # 1 step per vector env
            next_states, next_done, info = agent.rollout(next_states, next_done, step)

            if info != {}:
                if 'final_info' in info.keys():
                    infos = info['final_info']
                    for env_info in infos:
                        if env_info:
                            print(env_info['episode']['r'][0])
                            wandb.log({"episodic_return": env_info['episode']['r'][0]})
                            wandb.log({"episodic_length": env_info['episode']['l'][0]})

        # debug variables
        old_approx_kl, batch_values, batch_returns, v_loss, pg_loss, entropy_loss = agent.learn(envs, next_states)

        y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)

        # says if value function is a good indicator of returns
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        wandb.log({"losses/value_loss": v_loss.item()})
        wandb.log({"losses/policy_loss": pg_loss.item()})
        wandb.log({"losses/entropy": entropy_loss.item()})
        wandb.log({"losses/old_approx_kl": old_approx_kl.item()})
        wandb.log({"losses/explained_variance": explained_var})
        print("SPS:", int(global_step / (time.time() - start_time)))
        wandb.log({"charts/SPS": int(global_step / (time.time() - start_time))})

    envs.close()
