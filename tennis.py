import gym

env = gym.make("ALE/Tennis-v5", render_mode="human", obs_type='ram')
env.action_space.seed(42)

observation, info = env.reset(seed=42)

# https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
tennisMem = dict(enemy_x=27,
                 enemy_y=25,
                 enemy_score=70,
                 ball_x=16,
                 ball_y=17,
                 player_x=26,
                 player_y=24,
                 player_score=69
                 )

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    print(observation[tennisMem['player_x']], observation[tennisMem['player_y']], observation[tennisMem['enemy_score']])

    if terminated or truncated:
        observation, info = env.reset()


env.close()
