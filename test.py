from utils import create_agent
from typing import Optional
import numpy as np
import yaml
from core.args import set_args

def main(yaml_path:Optional[str]):
    args = set_args(yaml_path=yaml_path)
    alg_name = args.alg_name
    agent = create_agent(alg_name, args, multi_env=False, load=True) # single env for evaluation

    # Evaluate the model
    results = []
    for _ in range(agent.cfg.eval_epochs):
        epoch_reward = 0
        s = agent.env.reset()[0]
        while True:
            s, a, reward, done, info = agent.step(agent.env, s, deterministic=True, mode="test")
            epoch_reward += reward.item()
            if done:
                break
        results.append(epoch_reward)
    rewards = np.array(results)
    agent.logger.info(f"{agent.cfg.alg_name} in {agent.cfg.env_name}, Average {agent.cfg.eval_epochs} reward {np.mean(rewards):.3f}, Standard deviation {np.std(rewards):.3f}")

if __name__ == "__main__":
    main("results/PPO/HalfCheetah-v5/recipe.yaml")  # Don't have to change this line, you can set yaml_path by command line yaml_path=<your_path>