import mlflow
import numpy as np
import torch
from ml.evolutionary_training.evolutionary_algorithm.NeuralNetwork_class import PredefinedModel
from ml.evolutionary_training.evolutionary_algorithm.EvolutionaryAgent_class import EvolutionaryAgent
import random

class EvolutionaryAlgorithm:
    def __init__(self, population_size, num_generations, mutation_rate, input_dim, output_dim, hidden_dim=128):
        """
        Initialise the evolutionary algorithm.
        :param population_size: Number of individuals in the population.
        :param num_generations: Number of generations to train.
        :param mutation_rate: Rate at which mutations occur.
        :param input_dim: Input dimensionality for the model.
        :param output_dim: Output dimensionality for the model.
        :param hidden_dim: Number of neurons in the hidden layers.
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = [
            PredefinedModel(input_dim, output_dim, hidden_dim) for _ in range(population_size)
        ]

    def mutate(self, model):
        """
        Apply mutations to a model's weights and biases.
        """
        mutated_model = model.copy()
        for param in mutated_model.parameters():
            mutation_mask = torch.rand_like(param) < self.mutation_rate
            param.data += mutation_mask * torch.normal(mean=0, std=0.1, size=param.shape).to(param.device)
        return mutated_model

    def evaluate_population(self, population, num_players, num_games, game_env_class, action_space_manager,
                            game_state_encoder):
        """
        Evaluate the fitness of each individual in the population by having them play against other models.
        :param population: List of models in the population.
        :param num_players: Number of players in each game.
        :param num_games: Number of games to evaluate fitness.
        :param game_env_class: The game environment class.
        :param action_space_manager: The ActionSpaceManager instance.
        :param game_state_encoder: The GameStateEncoder instance.
        :return: A list of fitness scores for each model.
        """
        num_individuals = len(population)
        fitness_scores = [0] * num_individuals

        # Generate random matchups
        matchups = [
            random.sample(range(num_individuals), num_players)
            for _ in range(num_games)
        ]

        for matchup in matchups:
            # Create agents for the selected models
            agents = [
                EvolutionaryAgent(population[idx], action_space_manager, game_state_encoder)
                for idx in matchup
            ]

            # Initialise the game environment
            game_env = game_env_class(num_players, agents=agents)

            # Play the game and collect scores
            final_state = game_env.play_game()

            # Collect scores for each player in the matchup
            for i, model_idx in enumerate(matchup):
                player_score = final_state.player_boards[i]['score']  # Extract score for player `i`
                fitness_scores[model_idx] += player_score

        # Normalise fitness scores by the number of games played
        fitness_scores = [score / num_games for score in fitness_scores]

        return fitness_scores

    def select_parents(self, population, fitness_scores):
        """
        Select parents based on fitness scores, handling cases where scores are negative.
        :param population: List of models in the population.
        :param fitness_scores: List of fitness scores.
        :return: Two selected parent models.
        """
        # Adjust fitness scores to ensure non-negativity
        min_fitness = min(fitness_scores)
        adjusted_fitness_scores = [
            score - min_fitness + 1e-6 for score in fitness_scores
        ]  # Add a small constant to avoid zero probabilities

        # Calculate probabilities for parent selection
        total_fitness = sum(adjusted_fitness_scores)
        probabilities = [score / total_fitness for score in adjusted_fitness_scores]

        # Select two parents based on adjusted probabilities
        return np.random.choice(population, size=2, p=probabilities, replace=False)

    def run(self, game_env_class, action_space_manager, game_state_encoder, num_players, num_games):
        """
        Run the evolutionary algorithm with MLflow logging.
        """
        with mlflow.start_run():
            mlflow.log_param("population_size", self.population_size)
            mlflow.log_param("num_generations", self.num_generations)
            mlflow.log_param("mutation_rate", self.mutation_rate)

            best_model = None
            best_fitness = -float('inf')

            for generation in range(self.num_generations):
                print(f"Generation {generation + 1}/{self.num_generations}")

                fitness_scores = self.evaluate_population(
                    self.population, num_players, num_games, game_env_class, action_space_manager, game_state_encoder
                )

                max_fitness = max(fitness_scores)
                avg_fitness = sum(fitness_scores) / len(fitness_scores)

                mlflow.log_metric("max_fitness", max_fitness, step=generation)
                mlflow.log_metric("avg_fitness", avg_fitness, step=generation)

                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_model = self.population[np.argmax(fitness_scores)]

                new_population = []
                for _ in range(self.population_size):
                    parent1, parent2 = self.select_parents(self.population, fitness_scores)
                    child = self.mutate(parent1)
                    new_population.append(child)

                self.population = new_population
                print(f"Generation {generation + 1} - Best fitness: {max_fitness}, Avg fitness: {avg_fitness}")

            mlflow.log_metric("best_fitness", best_fitness)

            return best_model
