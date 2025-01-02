from ml.evolutionary_training.NEAT.train_neat_agents import train_neat_agents
from ml.evolutionary_training.evolutionary_algorithm.train_evolutionary_agents import train_evolutionary_agents
from game.GameVisualiser_class import GameVisualiser

from MinMax.MinMaxAgent_class import AzulAgent
from MinMax.MinMaxAzulEnv_class import MinMaxAzulEnv
from game.GameState_class import GameState

if __name__ == '__main__':
   # Example Usage
   game_state = GameState()

   game_state.reset()  # Set up initial game state

   # Create agents
   agents = [AzulAgent(player_idx=i) for i in range(game_state.num_players)]

   visualiser = GameVisualiser(num_players=game_state.num_players)

   # Set up the game environment
   env = MinMaxAzulEnv(game_state, agents, visualiser)

   # Simulate game
   final_scores, winner = env.play_game()
   print(f"Final Scores: {final_scores}")
   print(f"Winner: Player {winner + 1}")

# from game.GameState_class import GameState
# from game.GameVisualiser_class import GameVisualiser
#
# game_state = GameState()
# game_state.reset()
# game_state.take_action(0, 3, "red", 1)
# game_visualiser = GameVisualiser()
# game_visualiser.render(game_state)