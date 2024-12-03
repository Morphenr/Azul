import mlflow
import neat

class MLflowReporter(neat.reporting.BaseReporter):

    def __init__(self):
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        # Calculate average fitness
        fitnesses = [genome.fitness for genome in population.values()]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        # Log average fitness to MLflow
        mlflow.log_metric('average_fitness', avg_fitness, step=self.generation)

        # Number of species
        num_species = len(species_set.species)
        mlflow.log_metric('num_species', num_species, step=self.generation)

        # Average genome size (number of nodes and connections)
        total_nodes = []
        total_connections = []
        for genome in population.values():
            total_nodes.append(len(genome.nodes))
            total_connections.append(len(genome.connections))
        avg_nodes = sum(total_nodes) / len(total_nodes)
        avg_connections = sum(total_connections) / len(total_connections)
        avg_genome_size = avg_nodes + avg_connections

        # Log average genome size
        mlflow.log_metric('avg_num_nodes', avg_nodes, step=self.generation)
        mlflow.log_metric('avg_num_connections', avg_connections, step=self.generation)
        mlflow.log_metric('avg_genome_size', avg_genome_size, step=self.generation)

        # Optionally, log best fitness
        mlflow.log_metric('best_fitness', best_genome.fitness, step=self.generation)

    def end_generation(self, config, population, species_set):
        pass  # You can implement this if needed

    def complete_extinction(self):
        mlflow.log_metric('extinction_event', 1, step=self.generation)

    # Implement other methods if you wish to log more events