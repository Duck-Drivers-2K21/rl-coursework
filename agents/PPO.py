import time
import wandb
import gymnasium
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import torch.optim as optim
from wrappers import CroppedBorders, NoopsOnReset
import functools

POLICY_EVAL_INTERVAL_EPISODES = 10

# Code written following this tutorial: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/


def make_env(seed, idx, max_noops, num_frames_stack, frameskip, difficulty):
    def create_env():
        env = gymnasium.make("ALE/Pong-v5", difficulty=difficulty, render_mode="rgb_array", frameskip=frameskip,
                             repeat_action_probability=0)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env = CroppedBorders(env)
        if idx == 999:
            env = gymnasium.wrappers.RecordVideo(env, "ppo_videos", episode_trigger=lambda x: x % 1 == 0)
        env = NoopsOnReset(env, max_noops)
        env = gymnasium.wrappers.ResizeObservation(env, (84, 84))
        env = gymnasium.wrappers.GrayScaleObservation(env)
        env = gymnasium.wrappers.FrameStack(env, num_frames_stack)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return create_env


class Network(nn.Module):
    def __init__(self, num_frames, learning_rate):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            self.orthogonal_conv_layer_wrapper(num_frames, 32, 8, 4),
            nn.ReLU(),
            self.orthogonal_conv_layer_wrapper(32, 64, 4, 2),
            nn.ReLU(),
            self.orthogonal_conv_layer_wrapper(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            self.orthogonal_linear_layer_wrapper(3136, 512),
            nn.ReLU(),
        )

        # std of 0.01 ensure the output layers will have similar scalar values
        # making probability of taking each action to be similar
        self.actor = self.orthogonal_linear_layer_wrapper(512, envs.single_action_space.n, 0.01)
        self.critic = self.orthogonal_linear_layer_wrapper(512, 1, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    # pytorch by default uses different layer initialisation methods
    # therefore we must specify to be inline with PPO
    @staticmethod
    def orthogonal_conv_layer_wrapper(in_channels, out_channels, kernel_size, stride):
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride)
        torch.nn.init.orthogonal_(conv_layer.weight, np.sqrt(2))
        torch.nn.init.constant_(conv_layer.bias, 0)
        return conv_layer

    @staticmethod
    def orthogonal_linear_layer_wrapper(in_feat, out_feat, std=np.sqrt(2)):
        lin_layer = nn.Linear(in_features=in_feat, out_features=out_feat)
        torch.nn.init.orthogonal_(lin_layer.weight, std)
        torch.nn.init.constant_(lin_layer.bias, 0)
        return lin_layer

    def get_value(self, state):
        hidden = self.get_hidden_out(state)
        return self.critic(hidden)

    def get_hidden_out(self, state):
        return self.conv(state / 255)

    def get_action_and_value(self, state, action=None):
        hidden = self.get_hidden_out(state)
        logits = self.actor(hidden)  # unormalised action probs
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(hidden).flatten()
        return action, log_probs, entropy, value


class PPOAgent:

    def __init__(self, envs, args, device):
        self.args = args
        self.network = Network(self.args['num_frames_stack'],
                               self.args['learning_rate']).to(device)

        self.states = torch.zeros((self.args['num_steps'], self.args['num_env']) + envs.single_observation_space.shape) \
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
            action, logprob, _, self.values[step] = self.network.get_action_and_value(curr_states)

        self.logprobs[step] = logprob
        self.actions[step] = action

        new_next_states, reward, new_dones, _, new_info = envs.step(action.cpu().numpy())
        self.rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_states_tensor = torch.tensor(new_next_states).to(device)
        next_done_tensor = torch.tensor(new_dones, dtype=float).to(device)
        return next_states_tensor, next_done_tensor, new_info

    def calc_advantages(self, final_mask, final_val):
        with torch.no_grad():
            gae = torch.zeros_like(self.rewards).to(device)
            prev_gae = 0
            for time_step in reversed(range(self.args['num_steps'])):
                if time_step + 1 != self.args['num_steps']:
                    mask = 1 - self.dones[time_step + 1]
                    next_val = self.values[time_step + 1]
                else:
                    next_val = final_val
                    mask = final_mask
                delta = self.rewards[time_step] + self.args['gamma'] * next_val * mask - self.values[time_step]
                gae[time_step] = delta + self.args['gamma'] * self.args['gae_lambda'] * mask * prev_gae
                prev_gae = gae[time_step]
            returns = gae + self.values
        return gae.flatten(), returns.flatten()

    def generate_minibatches(self):
        batch_indices = np.arange(self.args['batch_size'])
        np.random.shuffle(batch_indices)
        batches = []
        for start in range(0, self.args['batch_size'], self.args['minibatch_size']):
            end = start + self.args['minibatch_size']
            minibatch_indicies = batch_indices[start:end]
            batches.append((start, end, minibatch_indicies))
        return batches

    def anneal_lr(self, update_num, num_updates):
        if update_num == num_updates:
            update_num -= 1
        prop_not_done = 1 - ((update_num) / num_updates)
        self.network.optimiser.param_groups[0]["lr"] = prop_not_done * args['learning_rate']

    def learn(self, envs, advantages, returns):
        # Advantage is the difference in predicted reward and actual value
        # Returns is what we get from the values predicted by the network

        # get stored state values
        logprobs = self.logprobs.flatten()
        states = self.states.reshape((-1,) + envs.single_observation_space.shape)
        actions = self.actions.flatten()

        batches = self.generate_minibatches()
        critic_loss, actor_loss, entropy_loss = 0, 0, 0
        for epoch in range(args['num_epochs']):

            # Use minibatches to optimise network
            for start, end, minibatch_indicies in batches:
                # forward pass into network
                _, newlogprob, entropy, new_val = self.network.get_action_and_value(
                    states[minibatch_indicies], actions.long()[minibatch_indicies]
                    # pass minibatch actions so agent doesnt sample new ones
                )



                logratio = newlogprob - logprobs[minibatch_indicies]
                ratio = logratio.exp()

                # probability clipping
                weighted_probs = advantages[minibatch_indicies] * ratio
                weighted_clipped_probs = torch.clamp(ratio, 1 - self.args['clip_co'], 1 + self.args['clip_co']) \
                                         * advantages[minibatch_indicies]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = ((new_val - returns[minibatch_indicies]) ** 2).mean()

                entropy_loss = entropy.mean()

                # larger entropy is larger exploration
                loss = actor_loss - self.args['ent_co'] * entropy_loss + critic_loss * self.args['vl_co']

                self.network.optimiser.zero_grad()
                loss.backward()
                self.network.optimiser.step()

            # returns debug variables for wandb
            return critic_loss, actor_loss


def evaluation_episode(env, agent):
    ep_return = 0
    state, info = env.reset()

    done = False
    while not done:
        with torch.no_grad():
            s = torch.Tensor(np.asarray(state)).to(device).unsqueeze(0)
            action, _, _, _ = agent.network.get_action_and_value(s)

        state, reward, done, trunc, info = env.step(action)
        ep_return += reward

    return ep_return


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(device)

    args = {
        "seed": 1,
        "learning_rate": 2.5e-4,
        'num_env': 8,
        'num_steps': 128,  # 2048
        'total_timesteps': 5000000,  # set arbitrarily high
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'num_minibatch': 4,
        'num_epochs': 4,
        'clip_co': 0.1,
        'ent_co': 0.01,
        'vl_co': 0.5,
        'max_noops': 30,
        'num_frames_stack': 4,
        'frameskip': 4,
        'difficulty': 0,
    }

    RUN_NAME = f"{args['num_env']}env_diff{args['difficulty']}_clip{args['clip_co']}"

    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    evaluation_episodic_return = []
    num_eps_processed = 0

    args['batch_size'] = args['num_env'] * args['num_steps']
    args['minibatch_size'] = int(args['num_steps'] // args['num_minibatch'])

    wandb.init(entity='cage-boys', project='final-final-runs', config=args, name=RUN_NAME)

    envs = gymnasium.vector.SyncVectorEnv(
        [make_env(args['seed'], i, args['max_noops'], args['num_frames_stack'], args['frameskip'], args['difficulty'])
         for i in
         range(args['num_env'])])

    eval_env = make_env(args['seed'], 999, args['max_noops'], args['num_frames_stack'], args['frameskip'],
                        args['difficulty'])()
    policy_evaluations_done = 0

    agent = PPOAgent(envs, args, device)

    steps_done = 0
    start_time = time.time()

    next_states = torch.tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args["num_env"]).to(device)
    num_updates = args['total_timesteps'] // args['batch_size']

    # training loop:

    # each update involves, interacting with env (for a num_steps), training policy and updating agent
    for update in range(1, num_updates + 1):

        agent.anneal_lr(update, num_updates)

        # policy rollout (doing num steps in env)
        # each policy step occurs in each vector env so we inc global step by num envs
        for step in range(0, args['num_steps']):
            steps_done += 1 * args['num_env']  # 1 step per vector env
            next_states, next_done, info = agent.rollout(next_states, next_done, step)

            if info != {}:
                if 'final_info' in info.keys():
                    infos = info['final_info']
                    for env_info in infos:
                        if env_info:
                            # print(env_info['episode']['r'][0])
                            wandb.log({"episodic_return": env_info['episode']['r'][0]}, step=steps_done)
                            wandb.log({"episodic_length": env_info['episode']['l'][0]}, step=steps_done)
                            evaluation_episodic_return.append(env_info['episode']['r'][0])
                            num_eps_processed += 1
                            if num_eps_processed % POLICY_EVAL_INTERVAL_EPISODES == 0 and len(
                                    evaluation_episodic_return) == POLICY_EVAL_INTERVAL_EPISODES:
                                total = functools.reduce(lambda a, b: a + b, evaluation_episodic_return)
                                wandb.log({"evaluation_episode_return": total / POLICY_EVAL_INTERVAL_EPISODES}, step=steps_done)
                                wandb.log({"policy_evaluation_episode_return": evaluation_episode(eval_env, agent)},
                                          step=steps_done)
                                wandb.log({"video": wandb.Video(
                                    f'ppo_videos/rl-video-episode-{policy_evaluations_done}.mp4', fps=30,
                                    format="mp4")})
                                policy_evaluations_done += 1
                                num_eps_processed = 0
                                evaluation_episodic_return = []

        advantages, returns = agent.calc_advantages(1 - next_done, agent.network.get_value(next_states).reshape(1, -1))
        # debug variables
        v_loss, pg_loss = agent.learn(envs, advantages, returns)

        # says if value function is a good indicator of returns
        wandb.log({"ppo/learning_rate": agent.network.optimiser.param_groups[0]["lr"]}, step=steps_done)
        wandb.log({"ppo/critic_loss": v_loss.item()}, step=steps_done)
        wandb.log({"ppo/actor_loss": pg_loss.item()}, step=steps_done)
        wandb.log({"steps_done": steps_done})
        wandb.log({"charts/SPS": int(steps_done / (time.time() - start_time))}, step=steps_done)

    envs.close()
