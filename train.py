import gymnasium as gym
from core.args import set_args
from typing import Optional
from utils import create_agent

def main(yaml_path:Optional[str]):
    args = set_args(yaml_path=yaml_path)
    agent = create_agent(args["alg_name"], args, multi_env=True, load=False)
    agent.train()
    agent.save()
    agent.monitor.learning_curve(mode="timestep")
    if args.episode_eval_freq is not None:
        agent.monitor.learning_curve(mode="episode")

if __name__ == "__main__":
    main(yaml_path="recipes/PPO.yaml") # Don't have to change this line, you can set yaml_path by command line yaml_path=<your_path>