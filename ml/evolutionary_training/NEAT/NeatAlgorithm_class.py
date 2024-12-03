import neat
import multiprocessing
import mlflow
import pickle
from ml.evolutionary_training.NEAT.evaluate_genome import evaluate_genome_pairs, create_genome_pairs
from ml.evolutionary_training.NEAT.MLFlowReporter_class import MLflowReporter
from helper_functions.helper_functions import load_game_settings
class NeatAlgorithm:
    def __init__(self, config, num_generations):
        self.config = config
        self.num_generations = num_generations
        self.population = neat.Population(self.config)
        # Add reporters for progress monitoring
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        # Add the custom MLflow reporter
        self.mlflow_reporter = MLflowReporter()
        self.population.add_reporter(self.mlflow_reporter)
        # Define maximum number of turns for a game
        self.max_turns = 1

    def run(self):
        # Start an MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("num_generations", self.num_generations)
            mlflow.log_param("population_size", self.config.pop_size)
            # Additional parameters can be logged here

            # Run NEAT
            winner = self.population.run(self.evaluate_genomes, n=self.num_generations)

            # Save the winner
            with open('best_genome.pkl', 'wb') as f:
                pickle.dump(winner, f)

            # Log the best genome
            mlflow.log_artifact('best_genome.pkl')

            return winner

    def evaluate_genomes(self, genomes, config):
        # Load dynamic game settings
        game_settings = load_game_settings()
        num_players = game_settings['num_players']
        num_games = 10  # Number of games per genome evaluation

        # Create a genome dictionary for easier access in multiprocessing
        genome_dict = {genome_id: genome for genome_id, genome in genomes}

        # Pair genomes for competition
        genome_pairs = create_genome_pairs(genomes, num_players)

        # Prepare multiprocessing arguments
        args = [
            (genome_ids, genome_dict, config, num_players, num_games, self.max_turns) for genome_ids in genome_pairs
        ]

        # Parallel evaluation of games
        with multiprocessing.Pool() as pool:
            results = pool.starmap(evaluate_genome_pairs, args)

        # Flatten results and aggregate scores
        scores = {genome_id: [] for genome_id, _ in genomes}
        for game_results in results:
            for genome_id, average_score in game_results:  # Unpack tuples
                scores[genome_id].append(average_score)

        # Assign fitness scores
        total_fitness = 0
        for genome_id, genome in genomes:
            genome.fitness = sum(scores[genome_id]) / len(scores[genome_id]) if scores[genome_id] else 0
            total_fitness += genome.fitness

        # Calculate the average fitness of the population
        avg_fitness = total_fitness / len(genomes)
        print(f"Average fitness: {avg_fitness}")

        if avg_fitness > 0:
            self.max_turns += 1
            print(f"Increasing max_turns to {self.max_turns}")
        else:
            self.max_turns = 1
            print(f"Resetting max_turns to {self.max_turns}")

        return avg_fitness