# RL-Lab

A repository for implementing reinforcement learning algorithms with streamlined visualization of learning curves.

## Project structure
```
RL-Lab/
├── core/               # core component
│   ├── args.py         # the arguments configuration
│   ├── module.py       # base RL class
│   ├── monitor.py      # monitoring the training process
│   ├── comparator.py   # used for comparing different algorithms
│   └── net.py          # nn architectures
├── models/             # supported RL algorithms
└── results/            # results with weight,learning_curve and hyperpatameters

```

## Train model
> [!IMPORTANT]
> These examples demonstrate minimal runnable instances. Check `core/args.py` for all available arguments that can be customized for each algorithm.

The basic command templates are:  (the unspecified parameter will adopt default value)
```shell
# Train via command line
python train.py --algname [name] --env_name [env_name] --max_timesteps [N]

# Train via YAML config
python train.py --algname [name] --config [recipe_path]
```

For example, :
```shell
# Train via command line
python train.py --alg_name REINFORCE --env_name CartPole-v1 --max_timesteps 20000

# Train via YAML config
python train.py --alg_name REINFORCE --config recipes/REINFORCE.yaml 
```

Then the model, recipe and learning curve will be saved at `results/`, and the learning curve looks like:

<img src="https://pic1.imgdb.cn/item/67eb916b0ba3d5a1d7e8fcf1" alt="timstep example" width="550" height="350">

The default `train_mode` is **timestep**. You can also train models from **episode** view, such as:

```
python main.py --alg_name REINFORCE --env_name CartPole-v1 --train_mode episode --max_epochs 300 
```

When switching from max_timesteps to max_epochs, the X-axis of the learning curve will change from timesteps to episodes accordingly.

<img src="https://pic1.imgdb.cn/item/67eb948e0ba3d5a1d7e8fe8b" alt="episode example" width="550" height="350">

> Obviously, the learning curve of episode trani_mode is not as smooth as timestep's, because as the agent interacts with the environment continues, subsequent rounds last longer. I strongly recommend using **timestep** train_mode.

The `alg_name` must be choosen from below:

 - **Q_Learning**
 - **Sarsa**
 - **REINFORCE**
 - **DQN Series**
   > For DQN series, you can combine modifiers like Double, Dueling, or Noisy with DQN at the end (e.g., DoubleDuelingDQN)
 - **A2C**
 - **PPO**

## Test model
You can test the model performance by:
```shell
# Train via result YAML config
python test.py --algname [name] --config [result_recipe_path]
```


## Compare Different Algorithms
You can modify the [compare.yaml](.recieps/compare.yaml), set the shared parameters that can be used by all algorithms, as well as set the hyperparameters specific to each algorithm.
```
python compare.py
```

<img src="https://pic1.imgdb.cn/item/67ebc7060ba3d5a1d7e91d32" alt="episode example" width="550" height="350">


The result will be saved at `result/comparison`. Your parameter settings will be automatically synchronized here for reproducing the results. Then you can modify the [compare.yaml](.recieps/compare.yaml) for another comparison experiment.

