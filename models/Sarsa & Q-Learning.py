import gymnasium as gym
from typing import Literal, Optional
import numpy as np
from module import VRL, VRLArgs

class Q_learning(VRL):
    def __init__(self, env, args):
        super().__init__(env, args=args)
        self.Q = np.zeros((self.state_num, self.action_num))
    
    def act(self, state, mode:Literal["train", "evaluate"]="train"):
        if mode == "train":
            a = self.epsilon_greedy(state)
        elif mode == "evaluate":
            a = np.argmax(self.Q[state])
        return a

    def train(self):
        for epoch in range(self.epochs):
            cur_s, _ = self.env.reset()
            epoch_reward = 0
            if self.alg_name == 'Q-Learning':
                while True:
                    self.sample_count += 1
                    a = self.act(cur_s)                      # behavior sugg，based on epsillon_greedy to choose the action.
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    epoch_reward += reward
                    self._update(cur_s, a, next_s, None, reward) # target sugg, using the best Q of s' to update
                    cur_s = next_s
                    if terminated or truncated:
                        break
            else:
                a = self.act(cur_s)  
                while True:
                    self.sample_count += 1
                    next_s, reward, terminated, truncated, info = self.env.step(a)
                    epoch_reward += reward
                    next_a = self.act(next_s)
                    self._update(cur_s, a, next_s, next_a, reward)    # performing TD by the actual action in s'
                    cur_s = next_s
                    a = next_a
                    if terminated or truncated:
                        break
            self.rewards_record.append(epoch_reward)

            if self.report(epoch=epoch):
                break 

    def _update(self, s, a, next_s, next_a:Optional[None], r):  
        if next_a == None:
            Q_target = r + self.gamma * np.max(self.Q[next_s])             
        else:
            Q_target = r + self.gamma * self.Q[next_s][next_a]                
        self.Q[s][a] = self.Q[s][a] + self.lr * (Q_target - self.Q[s][a])

if __name__ == "__main__":                      
    env = gym.make("CliffWalking-v0", render_mode="rgb_array", max_episode_steps=300)
    # args = VRLArgs(alg_name="Q-Learning", epochs=800, epsilon_start=0.3, epsilon_decay_flag=True, lr=0.4)
    args = VRLArgs(alg_name="Sarsa", epochs=800, epsilon_start=0.3, epsilon_decay_flag=True, lr=0.4)
    agent = Q_learning(env, args)
    agent.train()
    agent.learning_curve()

    # Increasing the lr of Q-learning, makes it easily converge to the optimal result, but it's risky when the env has amny local optimal solution.
    # Q-Learning --epochs 800 --epsilon_start 0.3 --epsilon_decay_flag --alpha 0.4 

    # Increasing the lr of Sarsa，which makes it significantly affects Q(s,a) if Q(s,a') falls into the cliff when the initial epsilon is large, thus effectively avoiding dangerous paths in the future.
    # Sarsa --epochs 800 --epsilon_start 0.2 --epsilon_decay_flag --alpha 0.4 --mode "Sarsa"   