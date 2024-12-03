from game.GameState_class import GameState
from helper_functions.GameStateEncoder_class import GameStateEncoder
from ml.ActionSpaceManager_class import ActionSpaceManager
import numpy as np

from helper_functions.helper_functions import get_valid_actions


game_state = GameState()
game_state.reset()

manager = ActionSpaceManager(game_state)


valid_actions = manager.get_valid_actions(game_state)
print("Valid Actions:", valid_actions)

valid_actions= get_valid_actions(game_state, game_state.current_player)
print("Valid Actions:", valid_actions)

activations = np.random.random(manager.action_space_size)

sorted_actions = manager.get_sorted_actions(activations, game_state)

print("Sorted Actions:", sorted_actions)