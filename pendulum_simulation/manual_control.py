import pygame
import pymunk
import pymunk.pygame_util
from commons import Pendulum, WIDTH, HEIGHT

# Configuration
FPS = 60

def main():
    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum Simulation - Manual Control")
    clock = pygame.time.Clock()
    
    # Set up physics space
    space = pymunk.Space()
    space.gravity = (0, 981)
    draw_options = pymunk.pygame_util.DrawOptions(window)
    
    # Create the pendulum
    pendulum = Pendulum(space)
    
    # Font for displaying information
    font = pygame.font.SysFont("Arial", 24)
    
    print("Simulation running... Press ESC or close window to exit.")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get sensory data from pendulum
        sensory_data = pendulum.get_sensory_data()
        
        # Use the keyboard to control the pendulum
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            move_speed = -5
        elif keys[pygame.K_RIGHT]:
            move_speed = 5
        else:
            move_speed = 0
        
        # Apply the movement to the pendulum pivot
        pendulum.pivot_body.velocity = (move_speed, 0)
        pendulum.pivot_body.position = (
            max(WIDTH/6, min(WIDTH - WIDTH/6, pendulum.pivot_body.position.x + move_speed)),
            pendulum.pivot_body.position.y
        )
        
        # Step the physics simulation
        dt = 1.0 / FPS
        space.step(dt)
        
        # Draw everything
        window.fill((240, 240, 240))  # Light gray background
        space.debug_draw(draw_options)
        
        # Display information
        pivot_x, angle, angular_vel = sensory_data
        info_text = [
            f"Pivot X: {pivot_x:.2f}",
            f"Angle: {angle:.2f}",
            f"Angular Velocity: {angular_vel:.2f}",
            f"Move Speed: {move_speed:.2f}",
            "Press ESC to exit"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            window.blit(text_surface, (10, 10 + i * 25))
        
        # Draw a reference for all keys at the bottom of the screen
        reference_font = pygame.font.SysFont('Arial', 16)
        keys_text = reference_font.render("Keys: ← → (Move Pivot)", 
                            True, (80, 80, 80))
        window.blit(keys_text, (20, HEIGHT - 30))
        
        pygame.display.update()
        clock.tick(FPS)
    
    pygame.quit()
    print("Simulation ended.")

if __name__ == "__main__":
    main()