import argparse
import pathlib
import torch
import cv2
import gymnasium as gym

import humanoid_bench
from .env import ROBOTS, TASKS
from hf_transformer.initTrafo import initialize_model
from hf_transformer.trainable_transformer import TrainableDT
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
    #import pdb; pdb.set_trace()
    print(f"ac_space = {env.action_space.shape}")
    # print("observation:", ob)
    env.render()
    ret = 0
    model=initialize_model(env.observation_space.shape[0], env.action_space.shape[0])
    #model=model.double()
    rew=torch.zeros((1,1),dtype=torch.float32)
    z=rew
    z=z.long()
    action=torch.zeros_like(torch.from_numpy(env.action_space.sample()))
    """"
    if isinstance(ob, dict):
        # Convert the values of ob into a numpy array
        ob_array = np.array(list(ob.values()))
    else:
        # Use ob as is
        ob_array = ob
    ob_array= ob_array.flatten()
    """
    #ob2 = torch.tensor(ob_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    #_, action, _ = model(states=ob, actions= action, rewards=rew, returns_to_go=rew, timesteps= z, attention_mask=rew, return_dict=False) #env.action_space.sample()
    #pdb.set_trace()
    #s needs to have shape torch.zeros([1, 1, 151]

    #see line 882 in https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L782

    s=torch.from_numpy(ob).float().unsqueeze(0).unsqueeze(0)
    print("shape",torch.from_numpy(env.action_space.sample()).view(1,1,-1).shape)
    _,a,_ = model(states=s, actions= action.view(1,1,-1), rewards=rew, returns_to_go=rew, timesteps= z, attention_mask=rew, return_dict=False)
    print("action",a.squeeze(0).squeeze(0).detach().numpy().shape)
    action=env.action_space.sample()
    i=0
    #print(torch.tensor(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    rew=0
    while True:
        #_, action, _ = model(states=ob, actions= action, rewards=rew, returns_to_go=rew, timesteps= z, attention_mask=rew, return_dict=False)
        #action = env.action_space.sample()
        #print("Iteration",i)
        #print("reward",rew)
        i+=1
        s=torch.from_numpy(ob).float().unsqueeze(0).unsqueeze(0)
        rew2= torch.tensor(rew, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        _,a2,_ = model(states=s, actions= torch.from_numpy(action).view(1,1,-1), rewards=rew2, returns_to_go=rew2, timesteps= z, attention_mask=rew2, return_dict=False)
        action = a2.squeeze(0).squeeze(0).detach().numpy()
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
