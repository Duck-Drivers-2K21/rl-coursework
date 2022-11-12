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

    @property
    def is_terminal(self): return self.terminated or self.truncated


class Environment:
    def __init__(self, seed: int = 42, render: bool = False):
        if render:
            env = gym.make("ALE/Pong-v5", render_mode="human", obs_type='ram')
        else:
            env = gym.make("ALE/Pong-v5", obs_type='ram')
        env.action_space.seed(seed)
        observation, info = env.reset(seed=seed)
        self.env = env
        self.state = State(observation, info)

    def step(self, action) -> float:
        if self.state.is_terminal:
            self.reset()  # Why are we resetting here? Wouldn't an assertion be more logical/safer?
        self.state.observation, reward, self.state.terminated, self.state.truncated, self.state.info = self.env.step(
            action)
        return reward

    def reset(self) -> None:
        observation, info = self.env.reset()
        self.state = State(observation, info)

    def close(self) -> None:
        self.env.close()
