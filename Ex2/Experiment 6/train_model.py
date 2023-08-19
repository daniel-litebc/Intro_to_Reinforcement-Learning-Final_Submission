from stable_baselines3 import PPO, DQN, A2C
from custom_env1 import *
import argparse
import os
import numpy as np

MODEL_DIR = "models/"
LOG_DIR = "logs"

if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

env = CustomSokoban2()
env.reset()

def main(model_num, model_version, time_steps=10000, num_iterations=50):


    model_path = MODEL_DIR + f"PPO-v{model_num}"

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    model.load(f"{model_path}/{model_version}")
    model.set_env(env)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    iter_num = len(os.listdir(model_path))

    for i in range(num_iterations):
        iter_num += 1
        model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=f"PPO-v{model_num}")
        model.save(f"{model_path}/{time_steps*iter_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
    parser.add_argument("model_num", type=str, help="Model type (e.g., 'PPO')")
    parser.add_argument("model_version", type=str, help="Name of the model")
    parser.add_argument("--time_steps", type=int, help="Number of time steps", default=10000)
    parser.add_argument("--num_iterations", type=int, help="Number of iterations", default=5)
    
    args = parser.parse_args()
    main(args.model_num, args.model_version, args.time_steps, args.num_iterations)