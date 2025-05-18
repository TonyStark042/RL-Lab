
from models.Q_Learning import Q_Learning
from models.PPO import PPO
from models.REINFORCE import REINFORCE
from models.A2C import A2C
from models.DQN_Series import DQN
from models.DDPG import DDPG
from models.TD3 import TD3
from models.SAC import SAC

MODEL_MAP = {
    "Q_Learning": Q_Learning,
    "Sarsa": Q_Learning,
    "REINFORCE": REINFORCE,
    "A2C": A2C,
    "PPO": PPO,
    "DQN": DQN,
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC,
}