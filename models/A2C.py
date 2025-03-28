from typing import Literal
from core.net import ActorCritic
from core.module import PRL, PRLArgs
from torch import optim
import torch
import gymnasium as gym
from argparse import ArgumentParser


class A2C(PRL):
    def __init__(self, env, args:PRLArgs) -> None:
        super().__init__(env, args=args, alg_name="A2C", model_name="model")
        self.model = ActorCritic(self.state_num, self.action_num, self.h_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state, mode:Literal["train", "evaluate"]="train"):
        state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
        if mode == "train":
            dist, value = self.model(state)
            action = dist.sample()
            return action.item(), value, dist
        else:
            dist, value = self.model(state)
            action = dist.probs.argmax()
            return action.item()
        
    def train(self):
        while self.epoch < self.max_epochs and self.timestep < self.max_timesteps:
            self.log_probs = []
            self.rewards = []
            self.epoch_values = []
            self.epoch_entropy = 0
            s = self.env.reset(seed=42)[0]

            while True:
                a, v, dist = self.act(s)
                s, reward, terminated, truncated, info = self.env.step(a)
                self.timestep += 1
                self.log_probs.append(dist.log_prob(torch.tensor(a, device=self.device)))
                self.rewards.append(reward)
                self.epoch_values.append(v)
                self.epoch_entropy += dist.entropy().mean()

                if self.timestep_freq:
                    early_stop = self.monitor.timestep_report()

                if terminated or truncated:
                    self.epoch_record.append(sum(self.rewards))
                    break

            self._update()

            if self.timestep_freq == None:
                early_stop = self.monitor.epoch_report()
                self.epoch += 1
            if early_stop:
                break

    def _update(self):
        returns = []
        for reward in self.rewards[::-1]:
            returns.insert(0, self.gamma * (0 if len(returns) ==0 else returns[0]) + reward)
        returns = torch.tensor(returns, device=self.device).detach()
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.epoch_values)
        advantage = returns - values  # To train the critic, be closer to the collcted return, so that better estimate the action advantage. 
        critic_loss = advantage.pow(2).mean()
        actor_loss  = -(log_probs * advantage.detach()).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.epoch_entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item()


if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="rgb_array", max_episode_steps=500)
    parser = ArgumentParser(description="A2C Settings")
    parser_args = parser.parse_args()
    alg_name = "A2C"
    args = PRLArgs( max_epochs=1000,
                    h_size=64, 
                    lr=0.001,
                    timestep_freq=100, 
                    max_timesteps=10000,
                    custom_args={
                                "sync_steps":64, 
                                "batch_size":32, 
                                "std_init":0.4,
                                })
    
    agent = A2C(env, args=args)
    agent.train()
    agent.learning_curve()
    agent.save()
    # save_dir = f"results/{alg_name}/{alg_name}_{env.spec.id}_h{agent.h_size}.pth"
    # agent.test(save_dir=save_dir)