import gymnasium as gym
from models.Q_Learning import Q_learning
from models.PPO import PPO
from models.REINFORCE import REINFORCE
from models.A2C import A2C
from models.DQN_Series import DQN
from core.args import *

def new_agent(**kwargs):
    """
    Create a new agent instance.
    """
    global args
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

def main():
    agent = new_agent() # using terminal to get args, do not need to pass args here

    if agent.mode == "train":
        agent.train()
        agent.save()
        agent.monitor.learning_curve(mode=agent.train_mode)
    elif agent.mode == "test":
        agent.test()

if __name__ == "__main__":
    main()