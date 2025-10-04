import pygame
import sys
import os
import subprocess

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
button_font = pygame.font.SysFont("Arial", 36)
info_font = pygame.font.SysFont("Arial", 24)

class Button:
    def __init__(self, x, y, width, height, text):
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
    # Create buttons
    train_button = Button(WIDTH//2 - 150, HEIGHT//2 - 50, 300, 80, "Train Neural Network")
    demo_button = Button(WIDTH//2 - 150, HEIGHT//2 + 50, 300, 80, "Run Demonstration")
    quit_button = Button(WIDTH//2 - 150, HEIGHT//2 + 150, 300, 80, "Quit")
    
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
                    subprocess.run([sys.executable, "pendulum_simulation/train.py"])
                except Exception as e:
                    print(f"Error running trainer: {e}")
                    
            if demo_button.is_clicked(mouse_pos, event):
                print("Starting demonstration...")
                try:
                    # Run the demonstration script
                    subprocess.run([sys.executable, "pendulum_simulation/demonstrate.py"])
                except Exception as e:
                    print(f"Error running demonstrator: {e}")
                    
            if quit_button.is_clicked(mouse_pos, event):
                running = False
        
        # Update button hover states
        train_button.check_hover(mouse_pos)
        demo_button.check_hover(mouse_pos)
        quit_button.check_hover(mouse_pos)
        
        # Draw everything
        screen.fill(BACKGROUND)
        
        # Draw title
        title_text = title_font.render("Pendulum Control", True, TITLE_COLOR)
        title_rect = title_text.get_rect(center=(WIDTH//2, 100))
        screen.blit(title_text, title_rect)
        
        # Draw buttons
        train_button.draw(screen)
        demo_button.draw(screen)
        quit_button.draw(screen)
        
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