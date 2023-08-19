import random
from utils.soko_pap import *
from collections import deque
import numpy as np
from gym import spaces


class CustomSokoban5(PushAndPullSokobanEnv):
    def __init__(self, dim_room=(7, 7), num_boxes=1, max_steps=500):
        super().__init__(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)
        obs = self.render()

        self.observation_space = spaces.Box(0, 255, obs.shape, obs.dtype)

    def reset(self):
        obs = super().reset()
        
        self.current_room_state = self.room_state.copy()
        self.distances = self.get_distances(self.current_room_state)
        
        return obs

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        self.next_room_state = self.room_state.copy()

        if not done:
            reward += self.reward_shaping(self.current_room_state, self.next_room_state, self.distances)

        self.current_room_state = self.next_room_state.copy()

        return next_state, reward, done, info

    def render(self, mode='rgb_array'):
        return super().render(mode=mode)


    def reward_shaping(self, current_room_state, next_room_state, distances_to_target):

        # If there is no change in the room state we are
        if np.array_equal(current_room_state, next_room_state): return -0.1

        current_box_position, current_monster_position = self.get_positions(current_room_state, distances_to_target)
        next_box_position, next_monster_position = self.get_positions(next_room_state, distances_to_target)

        # Calculate distances
        current_b2m = self.get_box_to_monster_distance(current_box_position, current_monster_position)
        next_b2m = self.get_box_to_monster_distance(next_box_position, next_monster_position)
        current_b2t = self.get_box_to_target_distance(current_box_position, distances_to_target)
        next_b2t = self.get_box_to_target_distance(next_box_position, distances_to_target)

        # Reward agent based on if box moved closer to target
        reward_b2t = 0.5 if next_b2t < current_b2t else (-0.5 if next_b2t > current_b2t else 0.0)

        # Reward agent based on if agent moved closer to box
        reward_b2m = 0.1 if (next_b2m < current_b2m and next_b2m >= 2) else (-0.1 if (next_b2m > current_b2m and next_b2m >= 2) else 0.0)

        return reward_b2t + reward_b2m + reward_adj_push


    def get_positions(self, room_state):
        box_position = tuple(np.argwhere(room_state == 4).ravel())
        monster_position = tuple(np.argwhere(room_state == 5).ravel())

        return box_position, monster_position


    def get_box_to_monster_distance(self, box_position, monster_position):
        return np.sum((np.array(monster_position) - np.array(box_position))**2)


    def get_box_to_target_distance(self, box_position, distances_to_target):
        return distances_to_target[box_position]


    def get_distances(self, room_state):
        target = np.where(room_state == 2)
        if target[0].size == 0:
            return None

        target = (target[0][0], target[1][0])
        distances = np.full_like(room_state, -1, dtype=int)
        distances[target] = 0

        visited_cells = {target}
        cell_queue = deque([target])

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while cell_queue:
            cell_x, cell_y = cell_queue.popleft()
            distance = distances[cell_x][cell_y]

            for dx, dy in directions:
                next_cell_x, next_cell_y = cell_x + dx, cell_y + dy

                if (0 <= next_cell_x < room_state.shape[0]
                        and 0 <= next_cell_y < room_state.shape[1]
                        and room_state[next_cell_x][next_cell_y] != 0
                        and (next_cell_x, next_cell_y) not in visited_cells):
                    distances[next_cell_x][next_cell_y] = distance + 1
                    visited_cells.add((next_cell_x, next_cell_y))
                    cell_queue.append((next_cell_x, next_cell_y))

        return distances
