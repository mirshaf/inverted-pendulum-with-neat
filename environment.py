import pygame
import pymunk
import pymunk.pygame_util
import math

pygame.init()

WIDTH, HEIGHT = 1000, 700
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Global variable to track if info should be displayed
show_info = False

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
        
        # Draw angular velocity vs time graph
        draw_angular_velocity_graph(window, angular_velocity_history)
    
    pygame.display.update()

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
    global show_info
    
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

        # Handle key presses to move the anchor
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            new_x = max(width/6, anchor_body.position.x - move_speed)
            anchor_body.position = (new_x, anchor_body.position.y)
            anchor_body.velocity = (-move_speed, 0)  # Set velocity for calculation
        elif keys[pygame.K_RIGHT]:
            new_x = min(width-width/6 , anchor_body.position.x + move_speed)
            anchor_body.position = (new_x, anchor_body.position.y)
            anchor_body.velocity = (move_speed, 0)  # Set velocity for calculation
        else:
            anchor_body.velocity = (0, 0)  # No movement

        draw(space, window, draw_options, anchor_body, circle_body, angular_velocity_history)
        space.step(dt)
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)
    
