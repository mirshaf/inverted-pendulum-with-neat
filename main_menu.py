import pygame
import sys
import subprocess
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pendulum Control - Main Menu")

# Colors
BACKGROUND = (240, 240, 240)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)
TEXT_COLOR = (255, 255, 255)
TITLE_COLOR = (50, 50, 50)

# Fonts
title_font = pygame.font.SysFont("Arial", 64, bold=True)
button_font = pygame.font.SysFont("Arial", 33)
info_font = pygame.font.SysFont("Arial", 24)

class Button:
    def __init__(self, text, x, y, width=300, height=72):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False
        
    def draw(self, surface):
        color = BUTTON_HOVER if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 3, border_radius=12)
        
        text_surf = button_font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

def main():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create buttons
    train_button = Button("Train Neural Network", WIDTH//2 - 150, HEIGHT//2 - 105)
    ai_control_button = Button("AI Control", WIDTH//2 - 150, HEIGHT//2 - 20)
    manual_control_button = Button("Manual Control", WIDTH//2 - 150, HEIGHT//2 + 65)
    quit_button = Button("Quit", WIDTH//2 - 150, HEIGHT//2 + 150)
    buttons = [train_button, ai_control_button, manual_control_button, quit_button]
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if train_button.is_clicked(mouse_pos, event):
                print("Starting training...")
                try:
                    # Run the training script
                    subprocess.run([sys.executable, os.path.join(current_script_dir, "pendulum_simulation", "train.py")])
                except Exception as e:
                    print(f"Error running trainer: {e}")
                    
            if ai_control_button.is_clicked(mouse_pos, event):
                print("Starting demonstration...")
                try:
                    # Run the demonstration script
                    subprocess.run([sys.executable, os.path.join(current_script_dir, "pendulum_simulation", "AI_control.py")])
                except Exception as e:
                    print(f"Error running demonstrator: {e}")
            
            if manual_control_button.is_clicked(mouse_pos, event):
                print("Starting demonstration...")
                try:
                    subprocess.run([sys.executable, os.path.join(current_script_dir, "pendulum_simulation", "manual_control.py")])
                except Exception as e:
                    print(f"Error running demonstrator: {e}")
                    
            if quit_button.is_clicked(mouse_pos, event):
                running = False
        
        # Update button hover states
        for b in buttons:
            b.check_hover(mouse_pos)
        
        # Draw everything
        screen.fill(BACKGROUND)
        
        # Draw title
        title_text = title_font.render("Pendulum Simulation", True, TITLE_COLOR)
        title_rect = title_text.get_rect(center=(WIDTH//2, 100))
        screen.blit(title_text, title_rect)
        
        # Draw buttons
        for b in buttons:
            b.draw(screen)
        
        # Draw info text
        info_text = info_font.render("Choose an option from the menu above", True, (100, 100, 100))
        info_rect = info_text.get_rect(center=(WIDTH//2, HEIGHT - 50))
        screen.blit(info_text, info_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()