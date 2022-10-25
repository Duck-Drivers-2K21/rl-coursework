import gym
import util


class State:
    def __init__(self, observation, info):
        self.observation = observation
        self.info = info
        self.terminated = False
        self.truncated = False

    @property
    def player_pos(self): return self._player_x, self._player_y

    @property
    def enemy_pos(self): return self._enemy_x, self._enemy_y

    @property
    def ball_pos(self): return self._ball_x, self._ball_y

    @property
    def _player_x(self): return self.observation[util.MemoryLocations.player_x]

    @property
    def _player_y(self): return self.observation[util.MemoryLocations.player_y]

    @property
    def _enemy_x(self): return self.observation[util.MemoryLocations.enemy_x]

    @property
    def _enemy_y(self): return self.observation[util.MemoryLocations.enemy_y]

    @property
    def enemy_score(self): return self.observation[util.MemoryLocations.enemy_score]

    @property
    def player_score(self): return self.observation[util.MemoryLocations.player_score]

    @property
    def _ball_x(self): return self.observation[util.MemoryLocations.ball_x]

    @property
    def _ball_y(self): return self.observation[util.MemoryLocations.ball_y]


class Environment:
    def __init__(self, seed: int = 42):
        env = gym.make("ALE/Tennis-v5", render_mode="human", obs_type='ram')
        env.action_space.seed(seed)
        observation, info = env.reset(seed=42)
        self.env = env
        self.state = State(observation, info)

    def step(self, action):
        if self.state.terminated or self.state.truncated:
            self.reset()
        self.state.observation, reward, self.state.terminated, self.state.truncated, self.state.info = self.env.step(
            action)
        return reward

    def reset(self):
        self.state.observation, self.state.info = self.env.reset()

    def close(self):
        return self.env.close()


def get_action(state: State):
    if state.player_x < state.ball_x:
        return util.Actions.DOWNRIGHTFIRE

    else:
        return util.Actions.DOWNLEFTFIRE


def run():
    env = Environment()
    action = util.Actions.NOOP

    try:
        while True:
            reward = env.step(action)
            action = get_action(env.state)
            print(reward)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        print("Closing environment")
        env.close()


if __name__ == "__main__":
    run()
