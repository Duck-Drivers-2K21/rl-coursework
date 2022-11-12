import util
import main
import random

# TODO: Do we want to add some form of abstract base-class for all agents?
#       Is something like this even possible? Is it worth the hustle?
# class Agent():
#   def __init__(self, env):
#     self.__env__ = env

#   def generate_episode(self):
#     pass

#   # Should we also add _get_action() here? Implementation detail...

class RandomAgent():
  def __init__(self, env : main.Environment):
    self.env = env

  def __get_action__(self) -> util.Actions:
    # Using env.action_space.sample() gives us the same outcome every time...
    # Opting to use random.randint(0, num_actions) instead
    assert util.Actions.NUM_ACTIONS == self.env.env.action_space.n, "Number of actions don't match..."
    return random.randint(0, util.Actions.NUM_ACTIONS - 1)

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
    env = main.Environment()
    agent = RandomAgent(env)
    try:
        _, rewards = agent.generate_episode()
        print(f"episode done in {len(rewards)} timesteps.")
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        print("Closing environment")
        env.close()


if __name__ == "__main__":
    run()
