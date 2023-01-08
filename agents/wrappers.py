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


class CroppedBorders(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        obs_shape = (160, 160) + env.observation_space.shape[2:]
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # Remove scores and cut unecessary whitespace...
        assert observation.shape == (210, 160, 3), "Sizes don't match..." + str(observation.shape)
        ROWS = [194 + j for j in range(16)] + [i for i in range(34)]
        cropped_frame = np.delete(observation, ROWS, axis=0)
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
