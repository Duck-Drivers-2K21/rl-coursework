import gymnasium

FRAME_SKIP = 4
DIFFICULTY = 0
ENV_SEED = 1

class MemoryLocations:
    # https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
    PLAYER_Y = 51
    PLAYER_X = 46
    ENEMY_Y = 50
    ENEMY_X = 45
    BALL_X = 49
    BALL_Y = 54
    ENEMY_SCORE = 13
    PLAYER_SCORE = 14


class Actions:
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5
    # Number of available actions:
    NUM_ACTIONS = 6


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
    def _player_y(self): return self.observation[MemoryLocations.PLAYER_Y]

    # @property
    # def _enemy_x(self): return self.observation[util.MemoryLocations.ENEMY_X]

    @property
    def _enemy_y(self): return self.observation[MemoryLocations.ENEMY_Y]

    @property
    def enemy_score(self): return self.observation[MemoryLocations.ENEMY_SCORE]

    @property
    def player_score(self): return self.observation[MemoryLocations.PLAYER_SCORE]

    @property
    def _ball_x(self): return self.observation[MemoryLocations.BALL_X]

    @property
    def _ball_y(self): return self.observation[MemoryLocations.BALL_Y]

    @property
    def is_terminal(self): return self.terminated or self.truncated


class NaiveEnvWrapper:
    """
    Environment Wrapper for naive agents. Uses RAM represnetation.
    """
    def __init__(self, seed: int = ENV_SEED, difficulty: int = DIFFICULTY, frame_skip=FRAME_SKIP, record: bool = False,
                 agent_name: str = "NoName"):
        env = gymnasium.make("ALE/Pong-v5", difficulty=difficulty, obs_type='ram', render_mode="rgb_array",
                             frameskip=frame_skip, repeat_action_probability=0)
        if record:
            env = gymnasium.wrappers.RecordVideo(env, "videos", episode_trigger=lambda x: x % 100 == 0, name_prefix=agent_name)

        env.action_space.seed(seed)
        observation, info = env.reset(seed=seed)
        self.env = env
        self.state = State(observation, info)

    def step(self, action: int) -> float:
        if self.state.is_terminal:
            assert False, "Terminal state in `step`."
        self.state.observation, reward, self.state.terminated, self.state.truncated, self.state.info = self.env.step(
            action)
        return reward

    def reset(self) -> None:
        observation, info = self.env.reset()
        self.state = State(observation, info)

    def close(self) -> None:
        self.env.close()
