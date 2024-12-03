import neat
from ml.evolutionary_training.NEAT.NeatAgent_class import NeatAgent
from ml.MultiAgentAzulEnv_class import MultiAgentAzulEnv
import random
import math

def evaluate_genome_pairs(genome_ids, genomes, config, num_players, num_games, max_turns):
    """
    Play multiple games with the same genome pairing and return averaged scores.
    """
    # Create agents for the current genome pairing
    genome_agents = [(genome_id, neat.nn.FeedForwardNetwork.create(genomes[genome_id], config)) for genome_id in genome_ids]

    total_scores = {genome_id: 0 for genome_id, _ in genome_agents}

    for _ in range(num_games):
        # Reset and play game
        env = MultiAgentAzulEnv(num_players=num_players)
        agents = [NeatAgent(net, env.action_space_manager, env.game_state_encoder) for genome_id, net in genome_agents]
        env.set_agents(agents)
        final_state = env.play_game(max_turns=max_turns)

        # Collect scores for each genome
        for idx, (genome_id, _) in enumerate(genome_agents):
            player_score = final_state.player_boards[idx]['score']
            total_scores[genome_id] += player_score

    # Calculate average scores
    average_scores = [(genome_id, total_score / num_games) for genome_id, total_score in total_scores.items()]
    return average_scores



def create_genome_pairs(genomes, num_players):
    """
    Create genome pairs ensuring every genome participates,
    even if the population size is not divisible by num_players.
    """
    genome_ids = [genome_id for genome_id, _ in genomes]
    random.shuffle(genome_ids)  # Shuffle to ensure random pairings

    # Wrap around if the population size is not divisible by num_players
    num_games = math.ceil(len(genome_ids) / num_players)  # Total games needed
    extended_genome_ids = genome_ids * num_games  # Wrap around genomes
    extended_genome_ids = extended_genome_ids[:num_games * num_players]  # Trim to exact size

    # Group into games of `num_players`
    genome_pairs = [extended_genome_ids[i:i + num_players] for i in range(0, len(extended_genome_ids), num_players)]
    return genome_pairs