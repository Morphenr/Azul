import mlflow
import numpy as np
from plotly.io import write_image
from ml.q_value.MultiAgentAzulEnv_class import MultiAgentAzulEnv
from ml.q_value.AzulAgent_class import AzulAgent
from helper_functions.helper_functions import encode_board_state, load_game_settings, encode_board_state
from helper_functions.plotting_functions import plot_metrics


def train_multi_agent(episodes=2000):
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

    # Prepare for logging metrics
    rewards_per_episode = []
    winning_scores = []
    episode_numbers = list(range(1, episodes + 1))

    # Start MLflow experiment
    with mlflow.start_run(run_name="Azul Multi-Agent Training"):
        print(f"Starting training for {episodes} episodes...\n")
        
        for episode in range(episodes):
            print(f"Starting Episode {episode + 1}...")
            
            # Play one complete game with the agents
            game_state = env.play_game()

            # Collect episode rewards and scores
            episode_rewards = [
                agent.current_step_reward for agent in agents
            ]  # Assuming agents track their rewards
            winning_score = max(player["score"] for player in env.game_state.player_boards)
            rewards_per_episode.append(np.max(episode_rewards))
            winning_scores.append(winning_score)

            print(f"Episode {episode + 1} complete. Game over. Collecting results...")
            print(f"Episode Average Reward: {np.mean(episode_rewards)}, Winning Score: {winning_score}")
            #print(f"Final game state: {env.game_state.__str__()}")

            # Log metrics to MLflow
            mlflow.log_metric("best_agent_reward", np.mean(episode_rewards), step=episode)
            mlflow.log_metric("winning_score", winning_score, step=episode)

        # Save model and log to MLflow
        for idx, agent in enumerate(agents):
            model_path = f"model_agent_{idx}.h5"
            agent.save_model(model_path)  # Assuming agents have a save_model method
            mlflow.log_artifact(model_path, artifact_path="models")

        # Generate and log performance plots
        print("Generating and saving performance plots...")
        fig = plot_metrics(rewards_per_episode, episode_numbers)
        fig_path = "average_rewards_plot.png"
        write_image(fig, fig_path)
        mlflow.log_artifact(fig_path, artifact_path="plots")

        # Finish training
        print(f"\nTraining complete. {episodes} episodes finished.")
