from ml.evolutionary_training.NEAT.train_neat_agents import train_neat_agents
from ml.evolutionary_training.evolutionary_algorithm.train_evolutionary_agents import train_evolutionary_agents

if __name__ == '__main__':
   train_neat_agents()

# from game.GameState_class import GameState
# from game.GameVisualiser_class import GameVisualiser
#
# game_state = GameState()
# game_state.reset()
# game_state.take_action(0, 3, "red", 1)
# game_visualiser = GameVisualiser()
# game_visualiser.render(game_state)