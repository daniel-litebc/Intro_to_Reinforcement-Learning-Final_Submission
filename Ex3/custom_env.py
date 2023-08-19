import random
from utils.soko_pap import *
from collections import deque
import numpy as np
from gym import spaces

class CustomSokobanEx3(PushAndPullSokobanEnv):
    def __init__(self, dim_room=(7, 7), num_boxes=2, max_steps=500):
        super().__init__(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)
        self.current_room_state = self.room_state.copy()
        self._states = self.get_states(self.current_room_state)

    def reset(self):
        obs = super().reset()
        self.current_room_state = self.room_state.copy()
        self._states = self.get_states(self.current_room_state)
        return obs

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self.next_room_state = self.room_state.copy()

        if not done:
            reward += self.reward_shaping(self.current_room_state, self.next_room_state, self.maze_info)

        self.current_room_state = self.next_room_state.copy()

        return next_state, reward, done, info

    def render(self, mode='rgb_array'):
        return super().render(mode=mode)

    def reward_shaping(self, current_room_state, next_room_state, maze_info):
        change_reward = self.box2target_change_reward(current_room_state, next_room_state, maze_info)
        return change_reward

    def get_positions(self, room_state):
        targets = [tuple(np.argwhere(room_state == 2)[i]) for i in range(2)]
        return targets

    def get_distances(self, room_state, target):
        return self.get_distances_for_target(room_state, target)

    def reward_shaping(self, current_room_state, next_room_state, maze_info):
        if np.array_equal(current_room_state, next_room_state):
            return -2.0

        targets = self.get_positions(current_room_state)
        distances0 = self.get_distances(current_room_state, targets[0])
        distances1 = self.get_distances(current_room_state, targets[1])
        common_distances = np.minimum(distances0, distances1)

        relevant_distances = common_distances

        if current_room_state[targets[0]] == 3:
            relevant_distances = distances1
        elif current_room_state[targets[1]] == 3:
            relevant_distances = distances0

        t2b = self.calc_distances(current_room_state, relevant_distances)
        n_t2b = self.calc_distances(next_room_state, relevant_distances)
        change_reward = 2.0 if n_t2b < t2b else (-2.0 if n_t2b > t2b else 0.0)

        return change_reward

    def get_states(self, current_room_state):
        targets = self.get_positions(current_room_state)
        distances0 = self.get_distances(current_room_state, targets[0])
        distances1 = self.get_distances(current_room_state, targets[1])
        common_distances = np.minimum(distances0, distances1)

        states = {
            'target0': targets[0],
            'target1': targets[1],
            'distances0': distances0,
            'distances1': distances1,
            'common_distances': common_distances
        }
        return states

    def get_distances(self, room_state, target):
        distances = np.zeros(shape=room_state.shape)
        visited_cells = set()
        cell_queue = deque()

        visited_cells.add(target)
        cell_queue.appendleft(target)

        while len(cell_queue) != 0:
            cell = cell_queue.pop()
            distance = distances[cell[0]][cell[1]]
            for x,y in ((1,0), (-1,-0), (0,1), (0,-1)):
                next_cell_x, next_cell_y = cell[0]+x, cell[1]+y
                if room_state[next_cell_x][next_cell_y] != 0 and not (next_cell_x, next_cell_y) in visited_cells:
                    distances[next_cell_x][next_cell_y] = distance + 1
                    visited_cells.add((next_cell_x, next_cell_y))
                    cell_queue.appendleft((next_cell_x, next_cell_y))

        return distances

    def calc_distances(self, room_state, distances_to_target):
        boxes = [tuple(np.argwhere(room_state == 4)[i]) for i in range(2)]
        if len(boxes) == 2:
            return distances_to_target[boxes[0]] + distances_to_target[boxes[1]]
        return distances_to_target[boxes[0]]