from core.args import VRLArgs, PRLArgs
from models import *
from core.args import args

def main():
    if args.max_episode_steps is None:
        env = gym.make(args.env_name, render_mode="rgb_array")
    else:
        env = gym.make(args.env_name, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)

    if args.alg_name in ["Q-Learning", "Sarsa"]:
        agent = Q_learning(env, args=args)
    elif args.alg_name == "PPO":
        agent = PPO(env, args=args)
    elif args.alg_name == "REINFORCE":
        agent = REINFORCE(env, args=args)
    elif args.alg_name == "A2C":
        agent = A2C(env, args=args)
    elif "DQN" in args.alg_name:
        agent = DQN(env, args=args)

    if args.mode == "train":
        agent.train()
        agent.save()
        agent.monitor.learning_curve(mode=args.train_mode)
    elif args.mode == "test":
        agent.test()

if __name__ == "__main__":
    main()