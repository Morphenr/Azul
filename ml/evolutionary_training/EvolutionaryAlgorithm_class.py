from ml.evolutionary_training.AzulAgent_class import AzulAgent
from ml.evolutionary_training.RandomAgent_class import RandomAgent
from ml.evolutionary_training.HeuristicAgent_class import HeuristicAgent
import random
import mlflow

class EvolutionaryAlgorithm:
    def __init__(self, pop_size, num_generations, input_dim, action_dim, env, mutation_rate=0.01):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.env = env
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_agent = None
        self.elite_agents = []
        self.max_turns = 1

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            agent = AzulAgent(self.input_dim, self.action_dim)
            self.population.append(agent)

    def evaluate_agent(self, agent, num_games=20):
        total_score = 0
        for _ in range(num_games):
            self.env.reset()
            # Decide opponent type
            opponent_type = random.choice(['random', 'elite'])
            #if opponent_type == 'heuristic':
            #    opponents = [HeuristicAgent()] * (self.env.num_players - 1)
            if opponent_type == 'random':
                opponent = RandomAgent()
                opponents = [opponent] * (self.env.num_players - 1)
            elif opponent_type == 'elite' and self.elite_agents:
                opponents = random.sample(self.elite_agents, self.env.num_players - 1)
            else:
                # Fallback to random opponents if elites are not available
                opponent = RandomAgent()
                opponents = [opponent] * (self.env.num_players - 1)
            agents = [agent] + opponents
            self.env.set_agents(agents)
            final_state = self.env.play_game(max_turns=self.max_turns)
            # Find the index of the agent being evaluated
            agent_index = agents.index(agent)
            player_score = final_state.player_boards[agent_index]['score']
            total_score += player_score
        average_score = total_score / num_games
        return average_score

    def evaluate_population(self):
        for agent in self.population:
            agent.fitness = self.evaluate_agent(agent)

    def select_parents(self, num_parents):
        # Sort agents by fitness
        sorted_population = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)
        parents = sorted_population[:num_parents]
        return parents

    def generate_next_population(self, parents):
        next_population = []
        # Elitism: Keep the best agent
        best_agent = parents[0].clone()
        next_population.append(best_agent)
        while len(next_population) < self.pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate)
            next_population.append(child)
        self.population = next_population

    def run(self):
        # Start an MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("population_size", self.pop_size)
            mlflow.log_param("num_generations", self.num_generations)
            mlflow.log_param("input_dim", self.input_dim)
            mlflow.log_param("action_dim", self.action_dim)
            mlflow.log_param("mutation_rate", self.mutation_rate)
            # You can add more parameters as needed

            # Initialize the population
            self.initialize_population()

            for generation in range(self.num_generations):
                print(f"Generation {generation + 1}")
                # Evaluate each agent
                self.evaluate_population()
                # Select parents
                num_parents = self.pop_size // 2
                parents = self.select_parents(num_parents)

                # Get best fitness and update best agent
                best_fitness = parents[0].fitness
                avg_fitness = sum(agent.fitness for agent in self.population) / len(self.population)
                print(f"Generation {generation + 1} - Best fitness: {best_fitness}, Average fitness: {avg_fitness}")

                if avg_fitness >= 0:
                    self.max_turns += 1

                # Update self.best_agent if necessary
                if self.best_agent is None or best_fitness > self.best_agent.fitness:
                    self.best_agent = parents[0].clone()

                # Update elite agents
                num_elites = min(10, len(parents))
                self.elite_agents = [parent.clone() for parent in parents[:num_elites]]

                # Log metrics
                mlflow.log_metric("best_fitness", best_fitness, step=generation)
                mlflow.log_metric("average_fitness", avg_fitness, step=generation)


                # Generate next generation
                self.generate_next_population(parents)

            # Log the best model
            mlflow.pytorch.log_model(self.best_agent.policy_network, "best_model")

            return self.best_agent
