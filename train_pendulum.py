import pygame
import pymunk
import pymunk.pygame_util
import math
import neat
import pickle  # Add this import

WIDTH, HEIGHT = 1000, 700
DRAW = False
total_sim_time = 30  # virtual simulation time in seconds
max_generations = 30

space = pymunk.Space()
space.gravity = (0, 981)

# generation = 0

class Pendulum:
    def __init__(self):
        # Create anchor body as KINEMATIC so we can control its position
        self.pivot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.pivot_body.position = (WIDTH/2, HEIGHT/2 - 50)

        self.bob_body = pymunk.Body() # The weighted object at the end of the pendulum
        self.bob_body.position = (WIDTH/2, HEIGHT/2 + 50)
        circle_shape = pymunk.Circle(self.bob_body, 20, (0, 0))
        circle_shape.friction = 1
        circle_shape.mass = 20
        circle_shape.elasticity = 0.95
        suspension = pymunk.PinJoint(self.bob_body, self.pivot_body, (0, 0), (0, 0)) # The cord that holds the bob and suspends it from a fixed point

        shape_filter = pymunk.ShapeFilter(group=1) # Make sure pendulums don't collide with each other
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
        if (pendulum_length == 0):
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
        
        # Normalize inputs for the neural net
        normalized_pivot_x = (self.pivot_body.position.x - WIDTH/6) / (WIDTH - WIDTH/3) * 2 - 1  # Map to [-1, 1]
        normalized_angle = angle / math.pi  # Map to [-1, 1]
        normalized_angular_velocity = self.get_angular_velocity() / math.pi  # Scale down
        
        return normalized_pivot_x, normalized_angle, normalized_angular_velocity
        

def fitness_function(population, config):
    """
    This function is solely for the use of the python NEAT library.

    Arguments:
        - The population as a list of (genome id, genome) tuples.
        - The current configuration object.

    The return value of the fitness function is ignored, but it must assign a Python float to the fitness member of each genome.
    The fitness function is free to maintain external state, perform evaluations in parallel, etc.
    It is assumed that fitness_function does not modify the list of genomes, the genomes themselves (apart from updating the fitness member), or the configuration object.
    """
    if DRAW:
        pygame.init() # Consider not renewing the pygame window each generation. 
        window = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        # generation_font = pygame.font.SysFont("Arial", 70)
        # font = pygame.font.SysFont("Arial", 30)
        draw_options = pymunk.pygame_util.DrawOptions(window)
    
    neural_nets = [] # Phenotypes
    genomes = []
    pendulums = []

    for genome_id, genome in population:
        neural_nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        pendulums.append(Pendulum())
        genome.fitness = 0
        genomes.append(genome)

    # Main game:

    # global generation
    # generation += 1
    
    fps = 60
    dt = 1 / fps

    number_of_steps = int(total_sim_time / dt)  # Total number of steps

    for _ in range(number_of_steps):
        if DRAW:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    import sys
                    sys.exit(0)
                
        pendulum: Pendulum
        for i, pendulum in enumerate(pendulums):
            # Take action:
            neural_net_outputs = neural_nets[i].activate(pendulum.get_sensory_data())
            move_speed = neural_net_outputs[0] * 5  # Speed at which the pendulum pivot moves

            pendulum.pivot_body.velocity = (move_speed, 0)
            pendulum.pivot_body.position = (
                max(WIDTH/6, min(WIDTH - WIDTH/6, pendulum.pivot_body.position.x + move_speed)),
                pendulum.pivot_body.position.y
            )

            # Adjust fitness: (should be after step?)
            # Threshold height (9/10 of pendulum length from anchor)
            threshold_height = pendulum.pivot_body.position.y - 0.9 * pendulum.pendulum_length
            # Check if pendulum is above threshold
            if pendulum.bob_body.position.y < threshold_height:
                genomes[i].fitness += dt
            # find and display best member?

        space.step(dt)

        if DRAW:
            window.fill((240, 240, 240))  # Light gray background
            space.debug_draw(draw_options)

            # text = generation_font.render("Generation : " + str(generation), True, (0, 0, 0))
            # text_rect = text.get_rect()
            # text_rect.center = (screen_width/2, screen_height/2 - 100)
            # screen.blit(text, text_rect)

            pygame.display.update()
            # clock.tick(fps) # Remove this for fast training

    for i, pendulum in enumerate(pendulums):
        space.remove(*pendulum.everything_in_space)
    # pygame.quit()

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(fitness_function, max_generations)
    
    # Save the winner genome to disk
    with open('best_pendulum_network.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
    
    print(f"\nBest network saved to 'best_pendulum_network.pkl'")
    print(f"Final fitness: {winner.fitness}")

if __name__ == "__main__":
    import os
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    run(config_path)