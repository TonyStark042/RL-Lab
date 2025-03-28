from core.args import VRLArgs
from models.Q_Learning import Q_learning
import gymnasium as gym

env = gym.make("CliffWalking-v0", render_mode="rgb_array", max_episode_steps=300)
# args = VRLArgs(alg_name="Q-Learning", max_epochs=800, epsilon_start=0.3, epsilon_decay_flag=True, lr=0.4)
args = VRLArgs( 
                # max_epochs=800, 
                epsilon_start=0.3, 
                epsilon_decay_flag=True, 
                lr=0.4,
                timestep_freq=100, 
                max_timesteps=10000,
                custom_args={"alg_name":"Sarsa"}
                )
agent = Q_learning(env, args)
agent.train()
agent.monitor.learning_curve(mode="timestep")