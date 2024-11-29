from ml.q_value.train_multi_agent import train_multi_agent
from ml.evolutionary_training.train_evolutionary_agents import train_evolutionary_agents
from ml.evolutionary_training.NEAT.train_neat_agents import train_neat_agents

# mlflow ui
if __name__ == '__main__':
    train_evolutionary_agents()