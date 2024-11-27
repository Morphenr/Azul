from ml.q_value.MultiAgentAzulEnv_class import MultiAgentAzulEnv
from ml.evolutionary_training.AzulAgent_class import AzulAgent
import numpy as np
import random

class EvolutionaryMultiAgentEnv:
    def __init__(self, num_agents, num_players, input_dim, action_dim):
        self.num_agents = num_agents
        self.num_players = num_players
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.population = self.initialize_population()
        self.env = MultiAgentAzulEnv(num_players=num_players)

    def initialize_population(self):
        """
        Initialize a population of agents with random weights.
        """
        return [AzulAgent(self.input_dim, self.action_dim, id=f"agent-{i}") for i in range(self.num_agents)]

    def evaluate_fitness(self, agent, num_episodes=1):
        """
        Evaluate an agent's fitness by playing games.
        """
        total_reward = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                valid_action_indices = self.env.get_valid_action_indices()
                q_values = agent.forward(state)
                q_values = self.env.mask_invalid_actions(q_values, valid_action_indices)
                action_index = np.argmax(q_values)  # Exploit
                action = self.env.game_state.get_action_space_mapper().index_to_action(action_index)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
        agent.fitness = total_reward
        return total_reward

    def select_parents(self, num_parents):
        """
        Select the top-performing agents to become parents.
        """
        sorted_agents = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        return sorted_agents[:num_parents]

    def evolve_population(self, num_parents, mutation_rate=0.1):
        """
        Perform selection, crossover, and mutation to create a new generation of agents.
        """
        parents = self.select_parents(num_parents)
        offspring = []

        while len(offspring) < self.num_agents - num_parents:
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate)
            offspring.append(child)

        # The new population consists of parents and their offspring
        self.population = parents + offspring
