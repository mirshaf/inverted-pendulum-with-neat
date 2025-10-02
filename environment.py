import pygame
import pymunk
import pymunk.pygame_util
import math
import random
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1000, 700
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Global variables
show_info = False
show_neural_net = False
neural_net_control = False  # False = keyboard control, True = neural net control

# Simple neural network class
class SimpleNeuralNet:
    def __init__(self):
        # Randomly initialize weights
        # Input to hidden: 3x2 matrix
        self.weights_ih = np.random.randn(3, 2) * 2 - 1  # Values between -1 and 1
        # Hidden to output: 2x1 matrix
        self.weights_ho = np.random.randn(2, 1) * 2 - 1  # Values between -1 and 1
        
        # Initialize activations for visualization
        self.input_activations = [0, 0, 0]
        self.hidden_activations = [0, 0]
        self.output_activation = 0
    
    def relu(self, x):
        return max(0, x)
    
    def tanh(self, x):
        return math.tanh(x)
    
    def forward(self, inputs):
        # Store input activations for visualization
        self.input_activations = inputs
        
        # Convert inputs to numpy array
        inputs = np.array(inputs).reshape(3, 1)
        
        # Calculate hidden layer activations
        hidden = np.dot(self.weights_ih.T, inputs)
        hidden = np.vectorize(self.relu)(hidden)
        
        # Store hidden activations for visualization
        self.hidden_activations = hidden.flatten().tolist()
        
        # Calculate output activation
        output = np.dot(self.weights_ho.T, hidden)
        output = self.tanh(output[0, 0])
        
        # Store output activation for visualization
        self.output_activation = output
        
        return output

# Create neural network instance
neural_net = SimpleNeuralNet()

def draw(space, window, draw_options, anchor_body, circle_body, angular_velocity_history):
    window.fill("white")
    space.debug_draw(draw_options)
    
    # Calculate the angle of the pendulum relative to positive x-axis
    dx = circle_body.position.x - anchor_body.position.x
    dy = circle_body.position.y - anchor_body.position.y
    angle = math.atan2(dy, dx)  # Returns angle in radians
    angle_degrees = math.degrees(angle)
    
    # Calculate the velocity of the circle relative to the anchor
    rel_velocity = (
        circle_body.velocity.x - anchor_body.velocity.x,
        circle_body.velocity.y - anchor_body.velocity.y
    )
    
    # Calculate tangential velocity (component perpendicular to the pendulum arm)
    pendulum_length = math.sqrt(dx**2 + dy**2)
    if pendulum_length > 0:
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
        angular_velocity = tangential_speed / pendulum_length
        angular_velocity_degrees = math.degrees(angular_velocity)
    else:
        angular_velocity_degrees = 0
    
    # Add current angular velocity to history (limit to 200 data points)
    angular_velocity_history.append(angular_velocity_degrees)
    if len(angular_velocity_history) > 200:
        angular_velocity_history.pop(0)
    
    # If neural net is controlling, get the move speed from the neural net
    move_speed = 0
    if neural_net_control:
        # Normalize inputs for the neural net
        normalized_anchor_x = (anchor_body.position.x - WIDTH/6) / (WIDTH - WIDTH/3) * 2 - 1  # Map to [-1, 1]
        normalized_angle = angle_degrees / 180  # Map to approximately [-1, 1] (since angle can be > 180)
        normalized_angular_velocity = angular_velocity_degrees / 100  # Scale down
        
        # Get output from neural net
        neural_net_output = neural_net.forward([normalized_anchor_x, normalized_angle, normalized_angular_velocity])
        move_speed = neural_net_output * 5  # Scale to appropriate move speed
    
    # Display information on screen if show_info is True
    if show_info:
        font = pygame.font.SysFont('Arial', 24)
        
        # Anchor x position
        anchor_text = font.render(f"Anchor X: {anchor_body.position.x:.2f}", True, (0, 0, 0))
        window.blit(anchor_text, (10, 10))
        
        # Pendulum angle
        angle_text = font.render(f"Angle: {angle_degrees:.2f}°", True, (0, 0, 0))
        window.blit(angle_text, (10, 40))
        
        # Angular velocity around anchor
        angular_velocity_text = font.render(f"Angular Velocity: {angular_velocity_degrees:.2f}°/s", True, (0, 0, 0))
        window.blit(angular_velocity_text, (10, 70))
        
        # Control mode
        control_text = font.render(f"Control: {'Neural Net' if neural_net_control else 'Keyboard'}", True, (0, 0, 0))
        window.blit(control_text, (10, 100))
        
        # Draw angular velocity vs time graph
        draw_angular_velocity_graph(window, angular_velocity_history)
    
    # Draw neural network visualization if enabled
    if show_neural_net:
        draw_neural_net(window)
    
    pygame.display.update()
    return move_speed

def draw_angular_velocity_graph(window, angular_velocity_history):
    if len(angular_velocity_history) < 2:
        return
    
    # Graph dimensions and position
    graph_width = 300
    graph_height = 200
    graph_x = WIDTH - graph_width - 20
    graph_y = 20
    
    # Draw graph background
    pygame.draw.rect(window, (240, 240, 240), (graph_x, graph_y, graph_width, graph_height))
    pygame.draw.rect(window, (0, 0, 0), (graph_x, graph_y, graph_width, graph_height), 1)
    
    # Find min and max values for scaling
    min_velocity = min(angular_velocity_history)
    max_velocity = max(angular_velocity_history)
    range_velocity = max_velocity - min_velocity
    
    # Avoid division by zero
    if range_velocity == 0:
        range_velocity = 1
    
    # Draw axes labels
    font = pygame.font.SysFont('Arial', 14)
    title = font.render("Angular Velocity vs Time", True, (0, 0, 0))
    window.blit(title, (graph_x + 10, graph_y + 5))
    
    min_label = font.render(f"{min_velocity:.1f}°/s", True, (0, 0, 0))
    window.blit(min_label, (graph_x - 40, graph_y + graph_height - 10))
    
    max_label = font.render(f"{max_velocity:.1f}°/s", True, (0, 0, 0))
    window.blit(max_label, (graph_x - 40, graph_y + 10))
    
    # Draw the graph line
    points = []
    for i, velocity in enumerate(angular_velocity_history):
        x = graph_x + (i / (len(angular_velocity_history) - 1)) * (graph_width - 1)
        y = graph_y + graph_height - ((velocity - min_velocity) / range_velocity) * (graph_height - 1)
        points.append((x, y))
    
    if len(points) > 1:
        pygame.draw.lines(window, (255, 0, 0), False, points, 2)

def draw_neural_net(window):
    # Neural network visualization parameters
    net_x = WIDTH - 350
    net_y = HEIGHT - 300
    layer_spacing = 150
    node_spacing = 60
    node_radius = 20
    
    # Draw connections (weights)
    for i in range(3):  # Input nodes
        for j in range(2):  # Hidden nodes
            weight = neural_net.weights_ih[i, j]
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)  # Green for positive, red for negative
            thickness = max(1, int(abs(weight) * 2))  # Thicker line for larger weights
            
            start_x = net_x
            start_y = net_y + i * node_spacing
            end_x = net_x + layer_spacing
            end_y = net_y + j * node_spacing
            
            pygame.draw.line(window, color, (start_x, start_y), (end_x, end_y), thickness)
    
    for j in range(2):  # Hidden nodes
        weight = neural_net.weights_ho[j, 0]
        color = (0, 255, 0) if weight > 0 else (255, 0, 0)  # Green for positive, red for negative
        thickness = max(1, int(abs(weight) * 2))  # Thicker line for larger weights
        
        start_x = net_x + layer_spacing
        start_y = net_y + j * node_spacing
        end_x = net_x + 2 * layer_spacing
        end_y = net_y + node_spacing  # Only one output node
        
        pygame.draw.line(window, color, (start_x, start_y), (end_x, end_y), thickness)
    
    # Draw input nodes
    for i in range(3):
        x = net_x
        y = net_y + i * node_spacing
        
        # Draw node outline
        pygame.draw.circle(window, (0, 0, 0), (x, y), node_radius, 2)
        
        # Draw activation indicator
        activation = neural_net.input_activations[i]
        activation_radius = int(min(node_radius, abs(activation) * node_radius / 3))
        activation_color = (0, 255, 0) if activation > 0 else (255, 0, 0)
        
        if activation_radius > 0:
            pygame.draw.circle(window, activation_color, (x, y), activation_radius)
        
        # Draw node label
        font = pygame.font.SysFont('Arial', 14)
        labels = ["Anchor X", "Angle", "Ang Vel"]
        label = font.render(labels[i], True, (0, 0, 0))
        window.blit(label, (x - 40, y - 40))
    
    # Draw hidden nodes
    for j in range(2):
        x = net_x + layer_spacing
        y = net_y + j * node_spacing
        
        # Draw node outline
        pygame.draw.circle(window, (0, 0, 0), (x, y), node_radius, 2)
        
        # Draw activation indicator
        activation = neural_net.hidden_activations[j]
        activation_radius = int(min(node_radius, abs(activation) * node_radius / 3))
        activation_color = (0, 255, 0) if activation > 0 else (255, 0, 0)
        
        if activation_radius > 0:
            pygame.draw.circle(window, activation_color, (x, y), activation_radius)
        
    
    # Draw output node
    x = net_x + 2 * layer_spacing
    y = net_y + node_spacing
    
    # Draw node outline
    pygame.draw.circle(window, (0, 0, 0), (x, y), node_radius, 2)
    
    # Draw activation indicator
    activation = neural_net.output_activation
    activation_radius = int(min(node_radius, abs(activation) * node_radius / 3))
    activation_color = (0, 255, 0) if activation > 0 else (255, 0, 0)
    
    if activation_radius > 0:
        pygame.draw.circle(window, activation_color, (x, y), activation_radius)
    
    # Draw node label
    font = pygame.font.SysFont('Arial', 14)
    label = font.render("Speed", True, (0, 0, 0))
    window.blit(label, (x - 40, y - 40))

def create_boundaries(space, width, height):
    rects = [
        [(width/2, height - 10), (width, 20)],
        [(width/2, 10), (width, 20)],
        [(10, height/2), (20, height)],
        [(width - 10, height/2), (20, height)]
    ]

    for pos, size in rects:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.4
        shape.friction = 0.5
        space.add(body, shape)

def create_pendulum(space, width, height):
    # Create anchor body as KINEMATIC so we can control its position
    anchor_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    anchor_body.position = (width/2, height/2 - 50)

    circle_body = pymunk.Body()
    circle_body.position = (width/2, height/2 + 50)
    circle_shape = pymunk.Circle(circle_body, 20, (0, 0))
    circle_shape.friction = 1
    circle_shape.mass = 40
    circle_shape.elasticity = 0.95
    rotation_center_joint = pymunk.PinJoint(circle_body, anchor_body, (0, 0), (0, 0))
    space.add(circle_shape, circle_body, rotation_center_joint)
    
    return anchor_body, circle_body  # Return both bodies for calculations

def run(window, width, height):
    global show_info, show_neural_net, neural_net_control
    
    run = True
    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps
    move_speed = 5  # Speed at which the anchor moves

    space = pymunk.Space()
    space.gravity = (0, 981)

    create_boundaries(space, width, height)
    anchor_body, circle_body = create_pendulum(space, width, height)  # Get both bodies

    draw_options = pymunk.pygame_util.DrawOptions(window)
    
    # History of angular velocity values for the graph
    angular_velocity_history = []

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    # Toggle info display when 'i' is pressed
                    show_info = not show_info
                elif event.key == pygame.K_n:
                    # Toggle neural net visualization when 'n' is pressed
                    show_neural_net = not show_neural_net
                elif event.key == pygame.K_c:
                    # Toggle control mode when 'c' is pressed
                    neural_net_control = not neural_net_control
                    # Reinitialize neural network with new random weights
                    neural_net = SimpleNeuralNet()

        # Get move speed from neural net if it's controlling
        neural_net_move_speed = draw(space, window, draw_options, anchor_body, circle_body, angular_velocity_history)
        
        # Handle movement based on control mode
        if neural_net_control:
            # Use neural net output to control the anchor
            anchor_body.velocity = (neural_net_move_speed, 0)
            anchor_body.position = (
                max(width/6, min(width - width/6, anchor_body.position.x + neural_net_move_speed)),
                anchor_body.position.y
            )
        else:
            # Keyboard control
            anchor_body.velocity = (0, 0)  # Reset velocity
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                new_x = max(width/6, anchor_body.position.x - move_speed)
                anchor_body.position = (new_x, anchor_body.position.y)
            elif keys[pygame.K_RIGHT]:
                new_x = min(width - width/6, anchor_body.position.x + move_speed)
                anchor_body.position = (new_x, anchor_body.position.y)

        space.step(dt)
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)
