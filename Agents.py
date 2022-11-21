import random
from abc import ABC, abstractmethod

from Pong import Environment, State, Actions


class Trajectory:
    """
    Wrapper for episodes generated by Naive Agents
    """

    def __init__(self) -> None:
        self.state_action_pairs = []  # [(state, action), ...]
        self.rewards = []  # [reward, ...]  Parallel with state_action_pairs.
        self.summed_undiscounted_rewards = 0  # Sum of undiscounted rewards.
        self.player_score = 0
        self.enemy_score = 0

    def append(self, state: State, action: int, reward: float) -> None:
        self.state_action_pairs.append((state, action))
        self.rewards.append(reward)
        self.summed_undiscounted_rewards += reward
        self.player_score = state.player_score
        self.enemy_score = state.enemy_score

    # Probably unnecessary but could make code cleaner later
    def __iadd__(self, other): self.append(*other)

    # replaces .timesteps
    def __len__(self) -> int: return len(self.rewards)


class AbstractAgent(ABC):
    """
    Abstract Class for Agents.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env

    @abstractmethod
    def generate_episode(self) -> Trajectory:
        """
        Returns the generated trajectory (state-action pairs, rewards, scores)
        """
        pass

    @property
    def name(self):
        return type(self).__name__

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.name}>"


class RandomAgent(AbstractAgent):
    """
    # Picks random action at each time-step.
    """

    def __init__(self, env: Environment) -> None:
        super().__init__(env)

    def _get_action(self) -> int:
        """

        """
        # Using env.action_space.sample() gives us the same outcome every time...
        # Opting to use random.randint(0, num_actions) instead.
        assert Actions.NUM_ACTIONS == self.env.env.action_space.n, "Number of actions don't match..."
        return random.randint(0, Actions.NUM_ACTIONS - 1)

    def generate_episode(self) -> Trajectory:
        trajectory = Trajectory()
        self.env.reset()

        while not self.env.state.is_terminal:
            state = self.env.state
            action = self._get_action()

            reward = self.env.step(action)

            trajectory.append(state, action, reward)
        return trajectory


class NaiveAgent(AbstractAgent):
    """
    Tracks the ball's coordinates and attempts to deflect it.
    """

    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self._tick = False
        self._prev_bx = 0

    def _get_action(self, state: State) -> int:
        px, py = state.player_pos
        bx, by = state.ball_pos
        bx, by = int(bx), int(by)

        bv = (bx - self._prev_bx)
        b_going_left = bv < 0
        bs = abs(bv)
        b_fast = (bs > 4)

        self._prev_bx = bx

        if by == 0:
            if abs(py - 80) < 10:
                return Actions.NOOP
            if py > 80:
                return Actions.RIGHTFIRE
            else:
                return Actions.LEFTFIRE

        self._tick = not self._tick

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
            return Actions.NOOP

        if d_y > 20 or self._tick or ball_close:
            if by < py_central:
                return Actions.RIGHTFIRE
            else:
                return Actions.LEFTFIRE

        return Actions.NOOP

    def generate_episode(self) -> Trajectory:
        trajectory = Trajectory()
        self.env.reset()

        self._tick = False
        self._prev_bx = 0

        while not self.env.state.is_terminal:
            state = self.env.state
            action = self._get_action(state)

            reward = self.env.step(action)

            trajectory.append(state, action, reward)
        return trajectory
