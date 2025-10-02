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
        body = pymunk.Body(body_type=pymunk.Body.STATIC)  # static bodies aren't affected by gravity
        body.position = pos
        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.4  # makes the surface bouncy, The object that we want to bounce must also have elasticity
        shape.friction = 0.5  # coefficient of friction
        space.add(body, shape)


def create_pendulum(space, width, height):
    anchor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    anchor_body.position = (width/2, height/2 - 100) # the anchor

    body = pymunk.Body()
    body.position = anchor_body.position # if not put on the anchor, will be connected to it with a line
    line = pymunk.Segment(body, (0, 0), (255, 0), 5) # body, relative pos to body 
    circle = pymunk.Circle(body, 40, (255, 0))
    line.friction = 1
    circle.friction = 1
    line.mass = 8
    circle.mass = 30
    circle.elasticity = 0.95
    rotation_center_joint = pymunk.PinJoint(body, anchor_body, (0, 0), (0, 0))
    space.add(circle, line, body, rotation_center_joint)

def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps

    space = pymunk.Space()
    space.gravity = (0, 981)

    create_boundaries(space, width, height)
    create_pendulum(space, width, height)

    draw_options = pymunk.pygame_util.DrawOptions(window)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        draw(space, window, draw_options)
        space.step(dt)  # how fast the simulation should go? I want to step 1/60th of a second in every iteration of the while loop
        clock.tick(fps)  # the while loop can run a maximum of 60 frames per second.

    pygame.quit()

if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)