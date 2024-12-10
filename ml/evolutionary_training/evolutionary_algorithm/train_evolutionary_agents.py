
from game.GameState_class import GameState
from helper_functions.GameStateEncoder_class import GameStateEncoder
from ml.ActionSpaceManager_class import ActionSpaceManager
from ml.evolutionary_training.evolutionary_algorithm.EvolutionaryAlgorithm_class import EvolutionaryAlgorithm
from ml.MultiAgentAzulEnv_class import MultiAgentAzulEnv
import mlflow

def train_evolutionary_agents():
    mlflow.set_experiment("Azul_Evolutionary_NN")

    # Determine input and output dimensions
    game_state = GameState()
    encoder = GameStateEncoder()
    features = encoder.encode(game_state)
    input_dim = len(features)
    manager = ActionSpaceManager(game_state)
    action_dim = manager.action_space_size

    # Evolutionary algorithm parameters
    population_size = 100
    num_generations = 10000
    mutation_rate = 0.1
    num_players = game_state.num_players
    num_games = 10

    # Initialise the evolutionary algorithm
    evolutionary_algorithm = EvolutionaryAlgorithm(
        population_size, num_generations, mutation_rate, input_dim, action_dim
    )

    # Train the population
    best_model = evolutionary_algorithm.run(
        MultiAgentAzulEnv, manager, encoder, num_players, num_games
    )

    print("Training complete")
