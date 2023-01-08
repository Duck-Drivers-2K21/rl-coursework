from typing import Type

from Pong import NaiveEnvWrapper
from BaselineAgents import RandomAgent, NaiveAgent, AbstractAgent


class AgentStats():
    def __init__(self, num_episodes: int, wins: int, total_undiscounted_rewards: int, total_timesteps: int) -> None:
        self.num_episodes = num_episodes
        self.wins = wins
        self.losses = self.num_episodes - wins
        self.win_ratio = wins / self.num_episodes
        self.average_undiscounted_rewards = total_undiscounted_rewards / self.num_episodes
        self.average_episode_length = total_timesteps / self.num_episodes

    def __str__(self) -> str:
        return f"Total Episodes: {self.num_episodes}. {self.wins} wins and {self.losses} losses (win ratio={self.win_ratio}).\n" \
               f"Average episodic undiscounted rewards {self.average_undiscounted_rewards}. Average episode length {self.average_episode_length}"


def train(agent_type: Type[AbstractAgent], n_episodes: int = 10) -> AgentStats:
    env = NaiveEnvWrapper(agent_name=agent_type.__name__)
    agent = agent_type(env)
    total_undiscounted_rewards = 0
    total_timesteps = 0
    wins = 0
    try:
        for _ in range(n_episodes):
            episode = agent.generate_episode()
            total_undiscounted_rewards += episode.summed_undiscounted_rewards
            total_timesteps += len(episode)
            if episode.player_score > episode.enemy_score:
                wins += 1
    except KeyboardInterrupt:
        print("Keyboard Interrupt...")
    finally:
        print("Closing environment.")
        env.close()
    return AgentStats(n_episodes, wins, total_undiscounted_rewards, total_timesteps)



if __name__ == "__main__":
    EPISODES = 1_000
    print("Training RandomAgent:")
    rand_stats = train(RandomAgent, EPISODES)
    print(rand_stats)
    print("---\nTraining NaiveAgent:")
    naive_stats = train(NaiveAgent, EPISODES)
    print(naive_stats)
