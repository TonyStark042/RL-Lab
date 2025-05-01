import gymnasium as gym
from models import MODEL_MAP
from utils import get_name
from core.args import ARGS_MAP

def main():
    alg_name = get_name()
    args_name = alg_name if "DQN" not in alg_name else "DQN"
    args = ARGS_MAP.get(args_name)(alg_name=alg_name)
    MODEL_CLASS = MODEL_MAP.get(args_name)

    env = gym.make(args.env_name, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    agent = MODEL_CLASS(env, args)
    agent.train()
    agent.save()
    agent.monitor.learning_curve(mode="timestep")
    if agent.episode_eval_freq is not None:
        agent.monitor.learning_curve(mode="episode")

if __name__ == "__main__":
    main()