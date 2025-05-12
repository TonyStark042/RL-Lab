from core.args import ARGS_MAP
allModels = list(ARGS_MAP.keys())
discrete_only_algorithms = ["DQN", "DoubleDuelingNoisyDQN", "Sarsa", "Q_Learning"]
continuous_algorithms = [algo for algo in allModels if algo not in discrete_only_algorithms]
noDeepLearning = ["Q_Learning", "Sarsa"]
VModels = ["Q_Learning", "Sarsa", "DQN", "DDPG", "TD3"]
PModels = ["REINFORCE", "PPO", "A2C"]