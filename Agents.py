from util import Actions
from main import Environment
import random

class RandomAgent():
  def __init__(self, env : Environment) -> None:
    self.env = env

  def __get_action__(self) -> int:
    # Using env.action_space.sample() gives us the same outcome every time...
    # Opting to use random.randint(0, num_actions) instead.
    assert Actions.NUM_ACTIONS == self.env.env.action_space.n, "Number of actions don't match..."
    return random.randint(0, Actions.NUM_ACTIONS - 1)

  def generate_episode(self) -> list:
    # Returns the undiscounted rewards the state action pairs and associated rewards: tuple(list, list).
    state_action_pairs = []
    rewards = []

    self.env.reset()

    while not self.env.state.is_terminal:
      state = self.env.state
      action = self.__get_action__()

      reward = self.env.step(action)

      state_action_pairs.append((state, action))
      rewards.append(reward)

    return state_action_pairs, rewards


def run():
    env = Environment()
    agent = RandomAgent(env)
    try:
        EPISODES = 10
        for _ in range(EPISODES):
          _, rewards = agent.generate_episode()
          print(f"episode done in {len(rewards)} timesteps.")
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        print("Closing environment")
        env.close()


if __name__ == "__main__":
    run()
