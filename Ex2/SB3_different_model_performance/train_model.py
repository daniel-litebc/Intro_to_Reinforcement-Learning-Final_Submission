from stable_baselines3 import PPO, DQN, A2C
from custom_env import Ex1EnvWrapper
import argparse
import os

MODEL_DIR = "models/"
LOG_DIR = "logs"

if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

env = Ex1EnvWrapper()
env.reset()

models = {"PPO": PPO("MlpPolicy", env, device='cuda', verbose=1, tensorboard_log=LOG_DIR),
          "A2C": A2C("MlpPolicy", env, device='cuda', verbose=1, tensorboard_log=LOG_DIR),
          "DQN": DQN("MlpPolicy", env, device='cuda', verbose=1, tensorboard_log=LOG_DIR, buffer_size=10000)}

def main(model_type, model_name, model_location=None, load_model=False, time_steps=10000, num_iterations=5):

    if load_model:
        model = models[model_type].load(model_location)
    else:
        model = models[model_type]

    model_path = MODEL_DIR + model_name


    if not os.path.exists(model_path):
        os.makedirs(model_path)

    iter_num = number_of_items = len(os.listdir(model_path))

    for i in range(num_iterations):
        iter_num += 1
        model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=model_name)
        model.save(f"{model_path}/{time_steps*iter_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
    parser.add_argument("model_type", type=str, help="Model type (e.g., 'PPO')")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("--model_location", type=str, help="Location of the model to load", default=None)
    parser.add_argument("--load_model", type=bool, help="Whether to load the model", default=False)
    parser.add_argument("--time_steps", type=int, help="Number of time steps", default=10000)
    parser.add_argument("--num_iterations", type=int, help="Number of iterations", default=5)

    args = parser.parse_args()
    main(args.model_type, args.model_name, args.model_location, args.load_model, args.time_steps, args.num_iterations)