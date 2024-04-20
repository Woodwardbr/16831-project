import argparse
import pathlib
import torch
import cv2
import gymnasium as gym

import humanoid_bench
from .env import ROBOTS, TASKS
from hf_transformer.initTrafo import initialize_model
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HumanoidBench environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Test offscreen rendering
    print(f"Test offscreen mode...")
    env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    ob, _ = env.reset()

    img = env.render()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_env_img.png", rgb_img)

    # Test online rendering with interactive viewer
    print(f"Test onscreen mode...")
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    ob, _ = env.reset()
    if isinstance(ob, dict):
        print(f"ob_space = {env.observation_space}")
        print(f"ob = ")
        for k, v in ob.items():
            print(f"  {k}: {v.shape}")
            assert (
                v.shape == env.observation_space.spaces[k].shape
            ), f"{v.shape} != {env.observation_space.spaces[k].shape}"
        assert ob.keys() == env.observation_space.spaces.keys()
    else:
        print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
        assert env.observation_space.shape == ob.shape
    print(f"ac_space = {env.action_space.shape}")
    # print("observation:", ob)
    env.render()
    ret = 0
    model=initialize_model(env.action_space.shape[0])
    while True:
        if isinstance(ob, dict):
        # Convert the values of ob into a numpy array
            ob_array = np.array(list(ob.values()))
        else:
        # Use ob as is
            ob_array = ob
        ob_tensor = torch.from_numpy(ob_array)
        ob2 = ob_tensor.view(1, -1)
        action = model(ob2) #env.action_space.sample()
        ob, rew, terminated, truncated, info = env.step(action)
        img = env.render()
        ret += rew

        if args.render_mode == "rgb_array":
            cv2.imshow("test_env", img[:, :, ::-1])
            cv2.waitKey(1)

        if terminated or truncated:
            ret = 0
            env.reset()
    env.close()
