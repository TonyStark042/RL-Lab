import yaml
from utils import new_agent
from core.comparator import Comparator
import torch.multiprocessing as mp
import pickle
mp.set_start_method('spawn', force=True)

if __name__ == "__main__":
    # com = pickle.load(open("results/comparisons/PPO_REINFORCE_A2C/CartPole-v1/comparator.pkl", "rb"))
    agents = []
    
    model_args = yaml.safe_load(open("compare.yaml", "r"))
    common_args = model_args.pop("common_args")
    com = Comparator(**common_args)

    for name, args in model_args.items():
        agent = new_agent(**args, **common_args, alg_name=name)
        com.add_algorithm((name, agent))

    com.train_all()
    com.learning_curve()
    com.save_all()