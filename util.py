class MemoryLocations:
    # https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
    player_y = 51
    player_x = 46
    enemy_y = 50
    enemy_x = 45
    ball_x = 49
    ball_y = 54
    enemy_score = 13
    player_score = 14


class Actions:
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5
