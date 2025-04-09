
from models.Q_Learning import Q_Learning
from models.PPO import PPO
from models.REINFORCE import REINFORCE
from models.A2C import A2C
from models.DQN_Series import DQN
from core.args import *

MODEL_MAP = {
    "Q_Learning": Q_Learning,
    "Sarsa": Q_Learning,
    "REINFORCE": REINFORCE,
    "A2C": A2C,
    "PPO": PPO,
    "DQN": DQN,
}