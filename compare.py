import yaml
from omegaconf import OmegaConf
from core.comparator import Comparator
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

if __name__ == "__main__":
    comparator = Comparator.initialize(recipe_path="recipes/compare.yaml")
    comparator.train_all()
    comparator.learning_curve()
    comparator.save_all()