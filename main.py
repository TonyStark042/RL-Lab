from models import *
from utils import new_agent

def main():
    agent = new_agent() # using terminal to get args, do not need to pass args here

    if agent.mode == "train":
        agent.train()
        agent.save()
        agent.monitor.learning_curve(mode=agent.train_mode)
    elif agent.mode == "test":
        agent.test()

if __name__ == "__main__":
    main()