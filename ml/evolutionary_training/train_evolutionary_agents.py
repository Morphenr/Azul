from game.GameState_class import GameState
from ml.evolutionary_training.EvolutionaryAlgorithm_class import EvolutionaryAlgorithm
from ml.evolutionary_training.MultiAgentAzulEnv_class import MultiAgentAzulEnv
from helper_functions.helper_functions import encode_board_state
import mlflow


def train_evolutionary_agents():

    mlflow.set_experiment('Azul_Evolutionary_Algorithm')

    # Determine input_dim and action_dim
    game_state = GameState()
    env = MultiAgentAzulEnv(num_players=game_state.num_players)
    input_dim = len(encode_board_state(game_state, 0)) # Generate the input_dim sizing
    action_dim = game_state.calculate_max_actions()

    print(f"Input dimension: {input_dim} - Action dimension: {action_dim}")

    pop_size = 50
    num_generations = 1000
    mutation_rate = 0.1  # You can adjust the mutation rate

    ea = EvolutionaryAlgorithm(
        pop_size=pop_size,
        num_generations=num_generations,
        input_dim=input_dim,
        action_dim=action_dim,
        env=env,
        mutation_rate=mutation_rate
    )

    best_agent = ea.run()

    # Save the best agent's model
    best_agent.save_model('best_azul_agent.pth')
