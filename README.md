# RL-Lab

A repository for implementing reinforcement learning algorithms with streamlined visualization of learning curves.

## Project structure
```
RL-Lab/
├── core/               # core component
│   ├── args.py         # the arguments configuration
│   ├── module.py       # base RL class
│   ├── monitor.py      # monitoring the training process
│   └── net.py          # nn architectures
├── models/             # supported RL algorithms
└── results/            # results with weight,learning_curve and hyperpatameters
```

## Examples
> [!IMPORTANT]
> These examples demonstrate minimal runnable instances. Check `core/args.py` for all available arguments that can be customized for each algorithm.

The basic command template is:

`python main --algname [name] --env [env_name] --max_timesteps [N]`

For example:

`python main.py --alg_name REINFORCE --env CartPole-v1 --max_timesteps 10000`

You can also train models from episode view, such as:

`python main.py --alg_name REINFORCE --env CartPole-v1 --max_epochs 300`

When switching from max_timesteps to max_epochs, the X-axis of the learning curve will change from timesteps to episodes accordingly.

The `alg_name` must be choosen from below:

 - **Q-Learning**
 - **Sarsa**
 - **REINFORCE**
 - **DQN Series**
   > For DQN variants, you can combine modifiers like Double, Dueling, or Noisy with DQN at the end (e.g., DoubleDuelingDQN)
 - **A2C**
 - **PPO**

