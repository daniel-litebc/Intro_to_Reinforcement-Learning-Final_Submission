import random
from utils.soko_pap import *

class Ex1EnvWrapper(PushAndPullSokobanEnv):
    def __init__(self, dim_room=(7,7), num_boxes=1, max_steps=500):
        super().__init__(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)

    def reset(self):
        obs = super().reset()
        return obs
    
    def step(self, action):
        return super().step(action)
    
    def render(self, mode='rgb_array'):
        return super().render(mode=mode)