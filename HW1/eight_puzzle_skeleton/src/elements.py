import pygame
from src.settings import *

"""
#############################################
    DO NOT CHANGE ANYTHING IN THIS FILE.
#############################################
"""

class Text:

    def __init__(self, x, y, text, text_size):
        """
            Initializes the text object class.

            Args:
                x (int): X coordinate of the text
                y (int): Y coordinate of the text
                text (str): Text to be displayed
                text_size (int): Size of the text      
        """
        pygame.font.init()
        self.x, self.y = x, y
        self.text = text
        self.text_size = text_size
    
    def draw(self, screen):
        """
            Draws the text on the screen.

            Args:
                screen (pygame.Surface): Screen to draw the text on
        """
        font = pygame.font.SysFont("Arial", self.text_size)
        text = font.render(self.text, True, WHITE)
        screen.blit(text, (self.x, self.y))

class Button:
    
    def __init__(self, x, y, width, height, text, text_size, background_color, text_color, radius = 0):
        """
            Initializes the button object class.

            Args:
                x (int): X coordinate of the button
                y (int): Y coordinate of the button
                width (int): Width of the button
                height (int): Height of the button
                text (str): Text to be displayed on the button
                text_size (int): Size of the text on the button
                background_color (tuple): Background color of the button
                text_color (tuple): Color of the text on the button
                radius (int): Radius of the button (default: 0)
        """
        pygame.font.init()
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.text = text
        self.text_size = text_size
        self.background_color = background_color
        self.text_color = text_color
        self.radius = radius

    def draw(self, screen):
        """
            Draws the button on the screen.

            Args:
                screen (pygame.Surface): Screen to draw the button on
        """
        pygame.draw.rect(screen, self.background_color, (self.x, self.y, self.width, self.height), border_radius = self.radius)
        self.font = pygame.font.SysFont("Arial", self.text_size)
        text = self.font.render(self.text, True, self.text_color)
        self.font_size = self.font.size(self.text)
        draw_x = self.x + (self.width - self.font_size[0]) // 2
        draw_y = self.y + (self.height - self.font_size[1]) // 2
        screen.blit(text, (draw_x, draw_y))
    
    def click(self, mouse_pos):
        """
            Checks if the button is clicked or not.
            
            Args:
                mouse_pos (tuple): Mouse position

            Returns:
                bool: True if the button is clicked, False otherwise
        """
        return (self.x <= mouse_pos[0] <= self.x + self.width) and (self.y <= mouse_pos[1] <= self.y + self.height)