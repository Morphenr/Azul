def generate_dynamic_config_file(input_dim, output_dim, file_path="config-feedforward.txt"):
    """
    Dynamically generate a NEAT configuration file with the specified input and output dimensions.

    Args:
        input_dim (int): Number of input nodes.
        output_dim (int): Number of output nodes.
        file_path (str): Path to save the generated configuration file.
    """
    config_content = f"""
[NEAT]
fitness_criterion     = mean
fitness_threshold     = 100
pop_size              = 200
reset_on_extinction   = True

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = relu sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 10.0
bias_min_value          = -10.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# Node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 10.0
response_min_value      = -10.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# genome compatibility options
compatibility_disjoint_coefficient = 2.0
compatibility_weight_coefficient   = 0.5

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 1
weight_mutate_rate      = 0.9
weight_replace_rate     = 0.1

# node gene options
node_add_prob = 0.35
node_delete_prob = 0.3
conn_add_prob = 0.4
conn_delete_prob = 0.35

# network parameters
num_hidden              = 0
num_inputs              = {input_dim}
num_outputs             = {output_dim}
feed_forward            = True
initial_connection      = unconnected

# genome configuration options
enabled_default         = True
enabled_mutate_rate     = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func    = mean
max_stagnation          = 10
species_elitism         = 2

[DefaultReproduction]
elitism                 = 1
survival_threshold      = 0.2
    """

    # Write the generated content to the specified file
    with open(file_path, "w") as file:
        file.write(config_content)
