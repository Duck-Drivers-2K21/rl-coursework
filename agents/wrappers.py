import gymnasium
import numpy as np

class NoopsOnReset(gymnasium.Wrapper):
    def __init__(self, env, max_noops):
        super(NoopsOnReset, self).__init__(env)
        self.max_noops = max_noops

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(np.random.randint(1, self.max_noops + 1)):
            obs, _, _, _, info = self.env.step(0)
        return obs, info


class CroppedBorders(gymnasium.Wrapper):
    def __init__(self, env):
        super(CroppedBorders, self).__init__(env)

    def _preprocess_frame(observation: np.ndarray) -> np.ndarray:
        # Remove scores and cut unecessary whitespace...
        assert observation.shape == (210, 160), "Sizes don't match..."
        ROWS = [194 + j for j in range(16)] + [i for i in range(34)]
        cropped_frame = np.delete(observation, ROWS, axis=0)
        # print(cropped_frame.shape)  # We've ended up with a 160x160 input...
        return cropped_frame


class RandomActionsOnReset(gymnasium.Wrapper):
    def __init__(self, env, max_random_actions):
        super(RandomActionsOnReset, self).__init__(env)
        self.max_random_actions = max_random_actions

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(np.random.randint(1, self.max_random_actions + 1)):
            obs = self.env.step(self.env.action_space.sample())[0]
        return obs, info
