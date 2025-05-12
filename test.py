from utils import create_agent
from typing import Optional
import numpy as np
import yaml


def main(yaml_path:Optional[str]):
    model_args = yaml.safe_load(open(yaml_path, "r"))
    alg_name = model_args["alg_name"]
    agent = create_agent(alg_name, model_args, multi_env=False, load=True) # single env for evaluation

    # Evaluate the model
    results = []
    for _ in range(agent.cfg.eval_epochs):
        epoch_reward = 0
        s = agent.env.reset()[0]
        while True:
            s, a, reward, done, info = agent.step(agent.env, s, deterministic=True)
            epoch_reward += reward.item()
            if done:
                break
        results.append(epoch_reward)
    rewards = np.array(results)
    agent.logger.info(f"{agent.cfg.alg_name} in {agent.cfg.env_name}, Average {agent.cfg.eval_epochs} reward {np.mean(rewards):.3f}, Standard deviation {np.std(rewards):.3f}")

if __name__ == "__main__":
    yaml_path = "results/PPO/HalfCheetah-v5/recipe.yaml"
    main(yaml_path)