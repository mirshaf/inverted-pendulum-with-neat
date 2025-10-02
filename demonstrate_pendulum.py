import pygame
import pymunk
import pymunk.pygame_util
import math
import pickle
import neat
import os

# Configuration
WIDTH, HEIGHT = 1000, 700
FPS = 60

class Pendulum:
    def __init__(self, space):
        # Create anchor body as KINEMATIC so we can control its position
        self.pivot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.pivot_body.position = (WIDTH/2, HEIGHT/2 - 50)

        self.bob_body = pymunk.Body()  # The weighted object at the end of the pendulum
        self.bob_body.position = (WIDTH/2, HEIGHT/2 + 50)
        circle_shape = pymunk.Circle(self.bob_body, 20, (0, 0))
        circle_shape.friction = 1
        circle_shape.mass = 20
        circle_shape.elasticity = 0.95
        suspension = pymunk.PinJoint(self.bob_body, self.pivot_body, (0, 0), (0, 0))

        shape_filter = pymunk.ShapeFilter(group=1)
        circle_shape.filter = shape_filter
        
        # Calculate pendulum length
        self.pendulum_length = math.sqrt(
            (self.bob_body.position.x - self.pivot_body.position.x)**2 + 
            (self.bob_body.position.y - self.pivot_body.position.y)**2
        )

        self.everything_in_space = [circle_shape, self.bob_body, suspension]
        space.add(*self.everything_in_space)

    def get_angular_velocity(self):
        # Calculate tangential velocity (component perpendicular to the pendulum arm)
        dx = self.bob_body.position.x - self.pivot_body.position.x
        dy = self.bob_body.position.y - self.pivot_body.position.y
        pendulum_length = math.sqrt(dx**2 + dy**2)
        if pendulum_length == 0:
            return 0

        # Calculate the velocity of the circle relative to the pivot
        rel_velocity = (
            self.bob_body.velocity.x - self.pivot_body.velocity.x,
            self.bob_body.velocity.y - self.pivot_body.velocity.y
        )
        
        # Unit vector along the pendulum arm
        arm_direction = (dx/pendulum_length, dy/pendulum_length)
        
        # Unit vector perpendicular to the pendulum arm (tangential direction)
        tangential_direction = (-arm_direction[1], arm_direction[0])
        
        # Dot product of velocity with tangential direction gives tangential speed
        tangential_speed = (
            rel_velocity[0] * tangential_direction[0] + 
            rel_velocity[1] * tangential_direction[1]
        )
        
        # Angular velocity = tangential speed / radius
        return tangential_speed / pendulum_length
    
    def get_sensory_data(self):
        # Calculate the angle of the pendulum relative to positive x-axis
        dx = self.bob_body.position.x - self.pivot_body.position.x
        dy = self.bob_body.position.y - self.pivot_body.position.y
        angle = math.atan2(dy, dx)  # Returns angle in radians
        
        # Normalize inputs for the neural net (same as in training)
        normalized_pivot_x = (self.pivot_body.position.x - WIDTH/6) / (WIDTH - WIDTH/3) * 2 - 1  # Map to [-1, 1]
        normalized_angle = angle / math.pi  # Map to [-1, 1]
        normalized_angular_velocity = self.get_angular_velocity() / math.pi  # Scale down
        
        return normalized_pivot_x, normalized_angle, normalized_angular_velocity

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
    local_dir = os.path.dirname(__file__)
    network_path = os.path.join(local_dir, 'best_pendulum_network.pkl')
    config_path = os.path.join(local_dir, 'neat_config.txt')
    
    if not os.path.exists(network_path):
        print(f"Error: Trained network file not found at {network_path}")
        print("Please run the training program first to generate best_pendulum_network.pkl")
        return
    
    try:
        network, winner_genome = load_trained_network(network_path, config_path)
    except Exception as e:
        print(f"Error loading network: {e}")
        return
    
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