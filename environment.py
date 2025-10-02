import pygame
import pymunk
import pymunk.pygame_util
import math

pygame.init()

WIDTH, HEIGHT = 1000, 700
window = pygame.display.set_mode((WIDTH, HEIGHT))

def draw(space, window, draw_options):
    window.fill("white")
    space.debug_draw(draw_options)
    pygame.display.update()

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
    anchor_body.position = (width/2, height/2 - 100)
    
    # Store the initial position for reference
    anchor_body.initial_position = (width/2, height/2 - 100)
    
    body = pymunk.Body()
    body.position = (width/2, height/2)
    line = pymunk.Segment(body, (0, 0), (255, 0), 5)
    circle = pymunk.Circle(body, 40, (255, 0))
    line.friction = 1
    circle.friction = 1
    line.mass = 8
    circle.mass = 30
    circle.elasticity = 0.95
    rotation_center_joint = pymunk.PinJoint(body, anchor_body, (0, 0), (0, 0))
    space.add(circle, line, body, rotation_center_joint)
    
    return anchor_body, body  # Return the anchor body so we can control it

def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps
    move_speed = 5  # Speed at which the anchor moves

    space = pymunk.Space()
    space.gravity = (0, 981)

    create_boundaries(space, width, height)
    anchor_body, pin = create_pendulum(space, width, height)  # Get the anchor body

    draw_options = pymunk.pygame_util.DrawOptions(window)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        # Handle key presses to move the anchor
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            anchor_body.position = (anchor_body.position.x - move_speed, anchor_body.position.y)
            # pin.position = (anchor_body.position.x - move_speed, anchor_body.position.y)
        if keys[pygame.K_RIGHT]:
            anchor_body.position = (anchor_body.position.x + move_speed, anchor_body.position.y)
            # pin.position = (anchor_body.position.x + move_speed, anchor_body.position.y)
        if keys[pygame.K_r]:  # Reset position with 'R' key
            anchor_body.position = anchor_body.initial_position
            pin.position = anchor_body.initial_position

        draw(space, window, draw_options)
        space.step(dt)
        clock.tick(fps)

    pygame.quit()

if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)
