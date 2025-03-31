from core.args import VRLArgs, PRLArgs
from models import *
from core.args import args
from utils import new_agent

def main():
    if args.max_episode_steps is None:
        env = gym.make(args.env_name, render_mode="rgb_array")
    else:
        env = gym.make(args.env_name, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)

    agent = new_agent(env, args=args)

    if args.mode == "train":
        agent.train()
        agent.save()
        agent.monitor.learning_curve(mode=args.train_mode)
    elif args.mode == "test":
        agent.test()

if __name__ == "__main__":
    main()