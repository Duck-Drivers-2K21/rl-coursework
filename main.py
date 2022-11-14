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
    def __init__(self, seed: int = 42, render: bool = False, difficulty=0):
        render_mode = "human" if render else None

        env = gym.make("ALE/Pong-v5", difficulty=difficulty, obs_type='ram', render_mode=render_mode)

        env.action_space.seed(seed)
        observation, info = env.reset(seed=seed)
        self.env = env
        self.state = State(observation, info)

    def step(self, action: int) -> float:
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


tick = False
prev_bx = 0


def get_action(state: State):
    global tick, prev_bx

    px, py = state.player_pos
    bx, by = state.ball_pos
    bx, by = int(bx), int(by)

    bv = (bx - prev_bx)
    b_going_left = bv < 0
    bs = abs(bv)
    b_fast = (bs > 4)

    # print(prev_bx, bx, bv, b_fast)

    prev_bx = bx

    if by == 0:
        if abs(py - 80) < 10:
            return util.Actions.NOOP
        if py > 80:
            return util.Actions.RIGHTFIRE
        else:
            return util.Actions.LEFTFIRE
    tick = not tick

    paddle_y_offset = 9

    close_distance = 20
    big_difference = 12
    small_difference = 8

    # if b_fast:
    #     small_difference = 5
    #     close_distance = 25

    ball_close = (bx >= px - close_distance)
    py_central = py + paddle_y_offset

    d_y = abs(by - py_central)

    if ((d_y < big_difference and not ball_close)
            or (d_y < small_difference)):
        return util.Actions.NOOP

    if d_y > 20 or tick or ball_close:
        if by < py_central:
            return util.Actions.RIGHTFIRE
        else:
            return util.Actions.LEFTFIRE

    return util.Actions.NOOP


def run():
    env = Environment()
    action = util.Actions.NOOP

    try:
        while True:
            reward = env.step(action)
            action = get_action(env.state)
            if env.state.terminated:
                player = env.state.observation[util.MemoryLocations.player_score]
                enemy = env.state.observation[util.MemoryLocations.enemy_score]
                win = player > enemy
                status = "Won" if win else "Lost"
                print(f"{status}: {player}, {enemy}")
                break
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    finally:
        print("Closing environment")
        env.close()

    def close(self) -> None:
        self.env.close()
