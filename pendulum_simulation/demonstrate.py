import pygame
import pymunk
import pymunk.pygame_util
import pickle
import neat
import os
from commons import Pendulum, WIDTH, HEIGHT

# Configuration
FPS = 60

def print_network(genome: neat.DefaultGenome):
    """
    Sources: 
        - https://neat-python.readthedocs.io/en/latest/_modules/genes.html
        - https://neat-python.readthedocs.io/en/latest/_modules/genome.html
    Node keys:
        - Input nodes: The key for an input node is the input's index plus one, then multiplied by negative one.
        - Output nodes: The key for an output node is simply its index within the list or tuple of outputs.
        - Hidden nodes: Hidden nodes are assigned positive integer keys that are unique within the genome.
    """
    print(f'Genome: key={genome.key} | fitness={genome.fitness}')
    print("Non-input nodes:")
    for key, value in genome.nodes.items():
        value: neat.genes.DefaultNodeGene
        print(f'    key={key}, bias={value.bias}, activation={value.activation}, aggregation={value.aggregation}')
    print("Connections:")
    for key, value in genome.connections.items():
        value: neat.genes.DefaultConnectionGene
        if value.enabled:
            print(f'    input_key={key[0]}, output_key={key[1]}, weight={value.weight}')
    
def load_trained_network(network_path, config_path):
    """Load the trained network from disk"""
    # Load the NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Load the winner genome
    with open(network_path, 'rb') as input_file:
        winner = pickle.load(input_file)
    
    # Create the neural network from the genome
    network = neat.nn.FeedForwardNetwork.create(winner, config)
    
    print(f"Loaded trained network with fitness: {winner.fitness}")
    return network, winner

def main():
    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trained Pendulum - Neural Network Control")
    clock = pygame.time.Clock()
    
    # Set up physics space
    space = pymunk.Space()
    space.gravity = (0, 981)
    draw_options = pymunk.pygame_util.DrawOptions(window)
    
    # Load the trained network
    local_dir = os.path.dirname(__file__) # This is always the same location (relative to your script)
    network_path = os.path.join(local_dir, 'best_network.pkl')
    config_path = os.path.join(local_dir, 'neat_config.txt')
    
    if not os.path.exists(network_path):
        print(f"Error: Trained network file not found at {network_path}")
        print("Please run the training program first to generate best_network.pkl")
        return
    
    try:
        network, winner_genome = load_trained_network(network_path, config_path)
    except Exception as e:
        print(f"Error loading network: {e}")
        return
    
    print_network(winner_genome)
    
    # Create the pendulum
    pendulum = Pendulum(space)
    
    # Font for displaying information
    font = pygame.font.SysFont("Arial", 24)
    
    print("Simulation running... Press ESC or close window to exit.")
    
    running = True
    while running:
        dt = 1.0 / FPS
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get sensory data from pendulum
        sensory_data = pendulum.get_sensory_data()
        
        # Use the neural network to control the pendulum
        neural_net_output = network.activate(sensory_data)
        move_speed = neural_net_output[0] * 5  # Same scaling as in training
        
        # Apply the movement to the pendulum pivot
        pendulum.pivot_body.velocity = (move_speed, 0)
        pendulum.pivot_body.position = (
            max(WIDTH/6, min(WIDTH - WIDTH/6, pendulum.pivot_body.position.x + move_speed)),
            pendulum.pivot_body.position.y
        )
        
        # Step the physics simulation
        space.step(dt)
        
        # Draw everything
        window.fill((240, 240, 240))  # Light gray background
        space.debug_draw(draw_options)
        
        # Display information
        pivot_x, angle, angular_vel = sensory_data
        info_text = [
            f"Fitness: {winner_genome.fitness:.2f}",
            f"Pivot X: {pivot_x:.2f}",
            f"Angle: {angle:.2f}",
            f"Angular Velocity: {angular_vel:.2f}",
            f"Move Speed: {move_speed:.2f}",
            "Press ESC to exit"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            window.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()
    print("Simulation ended.")

if __name__ == "__main__":
    main()