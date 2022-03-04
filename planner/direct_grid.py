import numpy as np

class DirectGrid(object):
    def __init__(self, env, scale):
        try:
            self.maze_low = env.env.maze_low
            maze_high = env.env.maze_high
        except:
            self.maze_low = env.env.initial_gripper_xpos[:2] - env.env.target_range
            maze_high = env.env.initial_gripper_xpos[:2] + env.env.target_range
        delta_scale = maze_high - self.maze_low
        x_num, y_num = int(delta_scale[0] // scale + 1), int(delta_scale[1] // scale + 1)
        self.scale = scale
        self.total_grid_num = x_num * y_num
        self.occupied_array = np.zeros((x_num, y_num))
        self.max_x = x_num
        self.max_y = y_num

    def update_occupied(self, positions):
        positions = positions - self.maze_low
        pos_indexs = (positions / self.scale).astype(int)
        x_index = pos_indexs[:, 0]
        y_index = pos_indexs[:, 1]
        if np.all(x_index >= 0) and np.all(x_index < self.max_x) and np.all(y_index >= 0) and np.all(y_index < self.max_y):
            self.occupied_array[x_index, y_index] += 1

    def coverage(self):
        not_zeros = np.where(self.occupied_array != 0)[0]
        return (len(not_zeros) / self.total_grid_num) / 7 * 9


