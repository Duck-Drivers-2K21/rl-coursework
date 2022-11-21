import gym
import util


class State:
    # Player and Enemy x-coordinates do not change. Paddles only move vertically.
    _ENEMY_X = 64
    _PLAYER_X = 188

    def __init__(self, observation, info):
        self.observation = observation
        self.info = info
        self.terminated = False
        self.truncated = False

    @property
    def player_pos(self): return self._PLAYER_X, self._player_y

    @property
    def enemy_pos(self): return self._ENEMY_X, self._enemy_y

    @property
    def ball_pos(self): return self._ball_x, self._ball_y

    # @property
    # def _player_x(self): return self.observation[util.MemoryLocations.PLAYER_X]

    @property
    def _player_y(self): return self.observation[util.MemoryLocations.PLAYER_Y]

    # @property
    # def _enemy_x(self): return self.observation[util.MemoryLocations.ENEMY_X]

    @property
    def _enemy_y(self): return self.observation[util.MemoryLocations.ENEMY_Y]

    @property
    def enemy_score(self): return self.observation[util.MemoryLocations.ENEMY_SCORE]

    @property
    def player_score(self): return self.observation[util.MemoryLocations.PLAYER_SCORE]

    @property
    def _ball_x(self): return self.observation[util.MemoryLocations.BALL_X]

    @property
    def _ball_y(self): return self.observation[util.MemoryLocations.BALL_Y]

    @property
    def is_terminal(self): return self.terminated or self.truncated


class Environment:
    def __init__(self, seed: int = 42, render: bool = False, difficulty: int = 0):
        render_mode = "human" if render else None

        env = gym.make("ALE/Pong-v5", difficulty=difficulty, obs_type='ram', render_mode=render_mode)

        env.action_space.seed(seed)
        observation, info = env.reset(seed=seed)
        self.env = env
        self.state = State(observation, info)

    def step(self, action: int) -> float:
        # This will need to be reworked for non-naive agents...
        if self.state.is_terminal:
            self.reset()  # TODO: Why are we resetting here? Wouldn't an assertion be more logical/safer?
        self.state.observation, reward, self.state.terminated, self.state.truncated, self.state.info = self.env.step(
            action)
        return reward

    def reset(self) -> None:
        observation, info = self.env.reset()
        self.state = State(observation, info)

    def close(self):
        return self.env.close()

