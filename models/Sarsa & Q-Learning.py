import gymnasium as gym
from typing import Literal, Optional
import numpy as np
import argparse
from matplotlib import pyplot as plt
import os

class Q_learning():
    def __init__(
        self,
        env,
        mode: Literal["Q-Learning", "Sarsa"] = "Q-Learning",
        gamma = 0.9,                     
        alpha = 0.01,                                  
        sample_count = 0,                
        epsilon_start = 0.1,             
        epsilon_end = 0.005,             
        # epsilon_decay = 0.0005,            
        epsilon_decay_flag = True        
    ):
        self.env = env
        self.mode = mode
        self.state_num = self.env.observation_space.n
        self.action_num = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.sample_count = sample_count
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.max_episode_steps = self.env.spec.max_episode_steps
        self.epsilon_decay = 1 / (self.max_episode_steps * 10)  # suggesting control in 1 / (max_steps_per_eos*10)，which means epsilon will be divided by e every 10 epochs.
        self.epsilon_decay_flag = epsilon_decay_flag 
        self.Q = np.zeros((self.state_num, self.action_num))
        self.epoch_rewards = {'actual':[], 'optimal':[]}
    
    def epsillon_greedy(self, state):
        if self.epsilon_decay_flag:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.sample_count *  self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_start  

        if np.random.rand() < self.epsilon:             
            return np.random.randint(self.action_num)
        else: 
            if np.count_nonzero(self.Q[state]) != 0: 
                return np.argmax(self.Q[state])
            else:
                return np.random.randint(self.action_num)
    
    def train(self, epochs):
        self.epochs = epochs
        for epoch in range(epochs):
            cur_s, _ = self.env.reset()
            epoch_reward = 0
            if self.mode == 'Q-Learning':
                while True:
                    self.sample_count += 1
                    a = self.epsillon_greedy(cur_s)                      # behavior sugg，based on epsillon_greedy to choose the action.
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    epoch_reward += reward
                    self._update_Q_table(cur_s, a, next_s, None, reward) # target sugg, using the best Q of s' to update
                    cur_s = next_s
                    if terminated or truncated:
                        break
            else:
                a = self.epsillon_greedy(cur_s)  
                while True:
                    self.sample_count += 1
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    epoch_reward += reward
                    next_a = self.epsillon_greedy(next_s)
                    self._update_Q_table(cur_s, a, next_s, next_a, reward)    # performing TD by the actual action in s'
                    cur_s = next_s
                    a = next_a
                    if terminated or truncated:
                        break
            
            optimal_reward = self._evaluate()
            self.epoch_rewards['optimal'].append(optimal_reward)                 
            self.epoch_rewards['actual'].append(epoch_reward)
            if epoch % 10 == 0:
                print(f"epoch: {epoch}, epoch_reward: {epoch_reward}, optimal_reward: {optimal_reward}")  

    def _update_Q_table(self, s, a, next_s, next_a:Optional[None], r):  
        if next_a == None:
            Q_target = r + self.gamma * np.max(self.Q[next_s])             
        else:
            Q_target = r + self.gamma * self.Q[next_s][next_a]                
        self.Q[s][a] = self.Q[s][a] + self.alpha * (Q_target - self.Q[s][a]) 

    # Execute the optimal strategy according to the Q-table; Use a 200 step cumulative reward to evaluate, if stop directly, it will be truncated, and if  don't converge, it will complete 200 steps
    def _evaluate(self):
        Q = self.Q
        env = self.env
        s, _ = env.reset()
        optimal_reward = 0.0 
        while True:           
            a = np.argmax(Q[s])                      
            s, reward, terminated, truncated, _ = env.step(a)  
            optimal_reward += reward                      
            if terminated or truncated:
                break
        return optimal_reward

    def learning_curve(self):
        file_path = __file__
        file_name = os.path.basename(file_path).split(".")[0]
        result_path = os.path.join("results", file_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

        x = range(self.epochs)
        y = self.epoch_rewards.get("optimal")
        plt.plot(x, y)
        name = self.mode + "_" + self.env.spec.id
        plt.title(name)
        plt.savefig(f'{os.path.join(result_path, name+".png")}', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--env', type=str, help='The experiment enviorment', default="CliffWalking-v0")
    parser.add_argument('--mode', type=str, help='The algorithm', default="Q-Learning")
    parser.add_argument('--epochs', type=int, help='The training epochs', default=500)
    parser.add_argument('--epsilon_start', type=float, help='The epsilon_start', default=0.2)
    parser.add_argument('--epsilon_end', type=float, help='The epsilon_end', default=0.005)
    parser.add_argument('--epsilon_decay_flag', action="store_true", help='Performing epsilon_decay or not', default=False)
    parser.add_argument('--alpha', type=float, help='The learning rate', default=0.1)
    parser.add_argument('--max_episode_steps', type=int, help='The maximum steps per epoch', default=200)
    args = parser.parse_args()
                        
    env = gym.make(args.env, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    agent = Q_learning(env, mode=args.mode, epsilon_start=args.epsilon_start, epsilon_decay_flag=args.epsilon_decay_flag, epsilon_end=args.epsilon_end, alpha=args.alpha)
    obs, info = env.reset()
    agent.train(epochs=args.epochs)
    # agent.learning_curve()

    # Increasing the lr of Q-learning, makes it easily converge to the optimal result, but it's risky when the env has amny local optimal solution.
    # Q-Learning --epochs 800 --epsilon_start 0.3 --epsilon_decay_flag --alpha 0.4 

    # Increasing the lr of Sarsa，which makes it significantly affects Q(s,a) if Q(s,a') falls into the cliff when the initial epsilon is large, thus effectively avoiding dangerous paths in the future.
    # Sarsa --epochs 800 --epsilon_start 0.2 --epsilon_decay_flag --alpha 0.4 --mode "Sarsa"   