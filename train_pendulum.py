import pygame
import pymunk
import pymunk.pygame_util
import neat
import pickle
from pendulum_commons import Pendulum, WIDTH, HEIGHT

DRAW = False
total_sim_time = 30  # virtual simulation time in seconds
max_generations = 10

space = pymunk.Space()
space.gravity = (0, 981)

# generation = 0      

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
        pendulums.append(Pendulum(space))
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

    winner: neat.DefaultGenome
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