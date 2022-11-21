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
