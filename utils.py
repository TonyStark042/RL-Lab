from core.args import *
import gymnasium as gym
from models import *

def new_agent(**kwargs):
    """
    Create a new agent instance.
    """
    if kwargs != {} and "alg_name" in kwargs:
        alg_name = kwargs.get("alg_name", None)
        if alg_name in ["Q-Learning", "Sarsa"]:
            args = VRLArgs(**kwargs)
        elif "PPO" in alg_name:
            args = PPOArgs(**kwargs)
        elif "REINFORCE" in alg_name:
            args = REINFORCEArgs(**kwargs)
        elif "A2C" in alg_name:
            args = A2CArgs(**kwargs)
        elif "DQN" in alg_name:
            args = DQNArgs(**kwargs)

    if args.max_episode_steps is None:
        env = gym.make(args.env_name, render_mode="rgb_array")
    else:
        env = gym.make(args.env_name, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)

    if args.alg_name in ["Q-Learning", "Sarsa"]:
        agent = Q_learning(env, args=args)
    elif "PPO" in args.alg_name:
        agent = PPO(env, args=args)
    elif "REINFORCE" in args.alg_name:
        agent = REINFORCE(env, args=args)
    elif "A2C" in args.alg_name:
        agent = A2C(env, args=args)
    elif "DQN" in args.alg_name:
        agent = DQN(env, args=args)
    return agent