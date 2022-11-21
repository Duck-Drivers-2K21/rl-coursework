from Pong import Environment
from Agents import RandomAgent, NaiveAgent


def train(agent_type: type, n_episodes: int = 10):
    env = Environment()
    agent = agent_type(env)
    try:
        for _ in range(n_episodes):
            episode = agent.generate_episode()
            outcome = "won" if episode.player_score > episode.enemy_score else "lost"
            print(
                f"Agent {outcome} with a score of {episode.player_score} to {episode.enemy_score}."
                f"({len(episode)} timesteps)"
            )
    except KeyboardInterrupt:
        print("Keyboard Interrupt...")
    finally:
        print("Closing environment.")
        env.close()


if __name__ == "__main__":
    print("Training RandomAgent:")
    train(RandomAgent)
    print("---\nTraining NaiveAgent:")
    train(NaiveAgent)
