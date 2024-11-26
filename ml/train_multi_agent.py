from ml.MultiAgentAzulEnv_class import MultiAgentAzulEnv
from ml.AzulAgent_class import AzulAgent
from helper_functions.helper_functions import encode_board_state, load_game_settings


def train_multi_agent(episodes=5000):
    # Load the game settings from the YAML configuration file
    print("Loading game settings...")
    settings = load_game_settings()
    num_players = settings.get('num_players')
    
    if num_players is None:
        print("Error: 'num_players' is not specified in the settings.")
        return

    print(f"Game Settings Loaded: {settings}")
    print(f"Initializing environment with {num_players} players...")
    
    # Initialize the MultiAgentAzulEnv with the specified number of players
    env = MultiAgentAzulEnv(num_players=num_players)

    # Encode the board state to determine input dimension
    print("Encoding board state to determine input dimension...")
    encoded_state = encode_board_state(env.game_state)
    print(f"Encoded State: {encoded_state}")
    input_dim = len(encoded_state)
    print(f"Input dimension determined: {input_dim} features in the encoded state.")

    # Retrieve the valid action space
    print("Retrieving size of action space")
    action_dim = len(env.game_state.get_action_space_mapper().index_to_action_map)
    print(f"Valid actions retrieved. Action dimension: {action_dim} possible actions.")

    # Initialize agents for each player based on the game settings
    print(f"Initializing {num_players} agents...")
    agents = [
        AzulAgent(input_dim=input_dim, action_dim=action_dim) for _ in range(num_players)
    ]
    env.set_agents(agents)
    print("Agents initialized and assigned to the environment.")

    # Training loop over the specified number of episodes
    print(f"Starting training for {episodes} episodes...\n")
    for episode in range(episodes):
        print(f"Starting Episode {episode + 1}...")

        # Play one complete game with the agents
        game_state = env.play_game()
        print(f"Episode {episode + 1} complete. Game over. Collecting results...")
        print(f"Final game state: {env.game_state.__str__()}")

    print(f"\nTraining complete. {episodes} episodes finished.")
