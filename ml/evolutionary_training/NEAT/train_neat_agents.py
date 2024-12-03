import os
import neat
import mlflow
import pickle
from game.GameState_class import GameState
from helper_functions.GameStateEncoder_class import GameStateEncoder
from ml.ActionSpaceManager_class import ActionSpaceManager
from ml.evolutionary_training.NEAT.NeatAlgorithm_class import NeatAlgorithm
from ml.evolutionary_training.NEAT.NeatAgent_class import NeatAgent
from ml.evolutionary_training.NEAT.generate_dynamic_config_file import generate_dynamic_config_file

def train_neat_agents():
    mlflow.set_experiment('Azul_NEAT_Algorithm')

    # Define the path to the NEAT config file
    config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward.txt')

    # Determine input_dim and action_dim
    game_state = GameState()
    encoder = GameStateEncoder()
    features = encoder.encode(game_state)
    input_dim = len(features)
    manager = ActionSpaceManager(game_state)
    action_dim = manager.action_space_size

    generate_dynamic_config_file(input_dim, action_dim, file_path=config_path)

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    #config.genome_config.input_keys = range(input_dim)
    #config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]

    print(f"Config - Inputs: {config.genome_config.num_inputs}, Outputs: {config.genome_config.num_outputs}")

    num_generations = 1000  # Adjust as needed

    # Initialize NEATAlgorithm
    neat_algorithm = NeatAlgorithm(config=config, num_generations=num_generations)

    # Run NEAT training
    winner = neat_algorithm.run()

    # Save the best genome
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    # Create a NEATAgent with the winner genome
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    best_agent = NeatAgent(net)

    # Save the best agent's model if needed
    # Implement a save_model method in NEATAgent if desired
    # best_agent.save_model('best_neat_agent.pkl')