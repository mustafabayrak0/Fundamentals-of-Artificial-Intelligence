import pygame
import random
import time

from src.settings import *
from src.tile import *
from src.elements import *

from agent.agent import *
from agent.agents import *


class Game:

    def __init__(self):
        """
            Initializes the game object class.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.shuffle_time = 0
        self.shuffle_move = SHUFFLE
        self.start_shuffle = False
        self.agent_activated = False
        self.prev_choice = ""
        self.start_game = False
        self.start_timer = False
        self.elapsed_time = 0
        self.high_score = self.get_score()
        self.image = "images/dog.jpg"

    def get_score(self):
        """
            Gets the high score from the score.txt file.

            Returns:
                float: High score
        """
        try:
            with open("score/score.txt", "r") as file:
                return float(file.read().splitlines()[0])
        except:
            return [0.000]

    def save_score(self):
        """
            Saves the high score to the score.txt file.
        """
        with open("score/score.txt", "w") as file:
            file.write(str("%.3f\n" % self.high_score))

    def create_game(self):
        """
            Creates the game matrix.
        """
        grid = [[x + y * GAMESIZE for x in range(1, GAMESIZE + 1)] for y in range(GAMESIZE)]
        grid[-1][-1] = 0
        return grid

    def draw_tiles(self):
        """
            Draws the tiles on the game grid.
        """
        self.tiles = []
        for row, x in enumerate(self.tiles_grid):
            self.tiles.append([])
            for col, y in enumerate(x):
                if y != 0:
                    self.tiles[row].append(Tile(self, col, row, str(y - 1), self.image))
                else:
                    self.tiles[row].append(Tile(self, col, row, "empty", self.image))

    def draw_grid(self):
        """
            Draws the game grid.
        """
        for row in range(-1, GAMESIZE * TILESIZE, TILESIZE):
            pygame.draw.line(self.screen, LIGHTGRAY, (START[0] + row, START[1]),
                             (START[0] + row, GAMESIZE * TILESIZE + START[1]))

        for column in range(-1, GAMESIZE * TILESIZE, TILESIZE):
            pygame.draw.line(self.screen, LIGHTGRAY, (START[0], START[1] + column),
                             (GAMESIZE * TILESIZE + START[0], START[1] + column))

    def draw_element(self):
        """
            Draws the UI elements.
        """
        for button in self.button_list:
            button.draw(self.screen)

        Text(825, 120, "Time", 30).draw(self.screen)
        Text(825, 160, "%.3f" % self.elapsed_time, 30).draw(self.screen)
        Text(100, 120, "High Score", 30).draw(self.screen)
        Text(130, 160, "%.3f" % (self.high_score if self.high_score > 0 else 0), 30).draw(self.screen)
        Text(810, 410, "Solve with", 30).draw(self.screen)

    def draw(self):
        """
            Game loop - draw.
        """
        self.screen.fill(BGCOLOUR)
        self.all_sprites.draw(self.screen)
        self.draw_grid()
        self.draw_element()
        pygame.display.flip()

    def new(self):
        """
            Starts a new game.
        """
        self.all_sprites = pygame.sprite.Group()
        self.tiles_grid = self.create_game()
        self.tiles_grid_completed = self.create_game()
        self.elapsed_time = 0
        self.agent_activated = False
        self.start_timer = False
        self.start_game = False
        self.draw_tiles()

        self.button_list = []
        self.button_list.append(Button(412, 600, 200, 50, "Shuffle", 25, WHITE, BLACK, 25))
        self.button_list.append(Button(412, 680, 200, 50, "Reset", 25, WHITE, BLACK, 25))

        self.button_list.append(Button(380, 510, 50, 50, "1", 25, WHITE, BLACK))
        self.button_list.append(Button(480, 510, 50, 50, "2", 25, WHITE, BLACK))
        self.button_list.append(Button(580, 510, 50, 50, "3", 25, WHITE, BLACK))

        self.button_list.append(Button(800, 460, 80, 50, "BFS T", 25, WHITE, BLACK, 10))
        self.button_list.append(Button(800, 515, 80, 50, "DFS T", 25, WHITE, BLACK, 10))
        self.button_list.append(Button(800, 570, 80, 50, "A* T", 25, WHITE, BLACK, 10))

        self.button_list.append(Button(900, 460, 80, 50, "BFS G", 25, WHITE, BLACK, 10))
        self.button_list.append(Button(900, 515, 80, 50, "DFS G", 25, WHITE, BLACK, 10))
        self.button_list.append(Button(900, 570, 80, 50, "A* G", 25, WHITE, BLACK, 10))

        self.button_list.append(Button(75, 550, 130, 40, "Shuffle: 5", 25, WHITE, BLACK, 10))
        self.button_list.append(Button(75, 600, 130, 40, "Shuffle: 10", 25, WHITE, BLACK, 10))
        self.button_list.append(Button(75, 650, 130, 40, "Shuffle: 15", 25, WHITE, BLACK, 10))

    def shuffle(self):
        """
            Shuffles the game matrix.
        """
        self.elapsed_time = 0
        possible_moves = []

        # Get all possible moves
        for row, tiles in enumerate(self.tiles):
            for col, tile in enumerate(tiles):
                if tile.text == "empty":
                    if tile.right():
                        possible_moves.append("right")
                    if tile.left():
                        possible_moves.append("left")
                    if tile.up():
                        possible_moves.append("up")
                    if tile.down():
                        possible_moves.append("down")
                    break
            if len(possible_moves) > 0:
                break

        # Removes the previously made move
        if self.prev_choice == "right":
            possible_moves.remove("left") if "left" in possible_moves else None
        elif self.prev_choice == "left":
            possible_moves.remove("right") if "right" in possible_moves else None
        elif self.prev_choice == "up":
            possible_moves.remove("down") if "down" in possible_moves else None
        elif self.prev_choice == "down":
            possible_moves.remove("up") if "up" in possible_moves else None

        # Selects a random move
        choice = random.choice(possible_moves)
        self.prev_choice = choice

        # Moves the tile
        if choice == "right":
            self.tiles_grid[row][col], self.tiles_grid[row][col + 1] = self.tiles_grid[row][col + 1], \
                                                                       self.tiles_grid[row][col]
        elif choice == "left":
            self.tiles_grid[row][col], self.tiles_grid[row][col - 1] = self.tiles_grid[row][col - 1], \
                                                                       self.tiles_grid[row][col]
        elif choice == "up":
            self.tiles_grid[row][col], self.tiles_grid[row - 1][col] = self.tiles_grid[row - 1][col], \
                                                                       self.tiles_grid[row][col]
        elif choice == "down":
            self.tiles_grid[row][col], self.tiles_grid[row + 1][col] = self.tiles_grid[row + 1][col], \
                                                                       self.tiles_grid[row][col]

    def solve(self):
        """
            Solves the game.
        """
        if len(self.moves) > 0:
            move = self.moves.pop(0)
            self.tiles_grid = move

    def events(self):
        """
            Game loop - events.
        """

        # Process input (events)
        for event in pygame.event.get():
            # Checks for closing window
            if event.type == pygame.QUIT:
                pygame.quit()
                quit(0)

            # Checks for mouse click and gets mouse position
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                # Switches clicked tile with the empty tile if not shuffling
                for row, tiles in enumerate(self.tiles):
                    for col, tile in enumerate(tiles):
                        if tile.click(mouse_pos) and not self.start_shuffle:

                            if tile.right() and self.tiles_grid[row][col + 1] == 0:
                                self.tiles_grid[row][col], self.tiles_grid[row][col + 1] = self.tiles_grid[row][
                                                                                               col + 1], \
                                                                                           self.tiles_grid[row][col]

                            if tile.left() and self.tiles_grid[row][col - 1] == 0:
                                self.tiles_grid[row][col], self.tiles_grid[row][col - 1] = self.tiles_grid[row][
                                                                                               col - 1], \
                                                                                           self.tiles_grid[row][col]

                            if tile.up() and self.tiles_grid[row - 1][col] == 0:
                                self.tiles_grid[row][col], self.tiles_grid[row - 1][col] = self.tiles_grid[row - 1][
                                                                                               col], \
                                                                                           self.tiles_grid[row][col]

                            if tile.down() and self.tiles_grid[row + 1][col] == 0:
                                self.tiles_grid[row][col], self.tiles_grid[row + 1][col] = self.tiles_grid[row + 1][
                                                                                               col], \
                                                                                           self.tiles_grid[row][col]

                            self.draw_tiles()

                # Checks for button click
                for button in self.button_list:
                    if button.click(mouse_pos):
                        if button.text == "Shuffle":
                            self.shuffle_time = 0
                            self.start_shuffle = True
                        if button.text == "Reset":
                            self.new()
                        if button.text == "1" or button.text == "2" or button.text == "3":
                            if button.text == "1":
                                self.image = "images/dog.jpg"
                            elif button.text == "2":
                                self.image = "images/cat.jpg"
                            elif button.text == "3":
                                self.image = "images/cub.jpg"
                            self.new()
                        if button.text == "BFS T" or button.text == "DFS T" or button.text == "A* T":
                            if button.text == "BFS T":
                                agent = BFSAgent(self.tiles_grid)
                            elif button.text == "DFS T":
                                agent = DFSAgent(self.tiles_grid)
                            elif button.text == "A* T":
                                agent = AStarAgent(self.tiles_grid)
                            self.agent_activated = True
                            self.moves = agent.tree_solve()
                            agent.print_info()
                        if button.text == "BFS G" or button.text == "DFS G" or button.text == "A* G":
                            if button.text == "BFS G":
                                agent = BFSAgent(self.tiles_grid)
                            elif button.text == "DFS G":
                                agent = DFSAgent(self.tiles_grid)
                            elif button.text == "A* G":
                                agent = AStarAgent(self.tiles_grid)
                            self.agent_activated = True
                            self.moves = agent.graph_solve()
                            agent.print_info()
                        if button.text == "Shuffle: 5" or button.text == "Shuffle: 10" or button.text == "Shuffle: 15":
                            if button.text == "Shuffle: 5":
                                self.shuffle_move = 5
                            elif button.text == "Shuffle: 10":
                                self.shuffle_move = 10
                            elif button.text == "Shuffle: 15":
                                self.shuffle_move = 15
                            self.shuffle_time = 0
                            self.start_shuffle = True

    def update(self):
        """
            Game loop - update.
        """
        # Checks if the game is over
        if self.start_game:
            if self.tiles_grid == self.tiles_grid_completed:
                self.start_game = False
                # Saves the high score
                if self.high_score > 0:
                    if self.elapsed_time < self.high_score:
                        self.high_score = self.elapsed_time
                    else:
                        self.high_score = self.high_score
                else:
                    self.high_score = self.elapsed_time
                self.save_score()

            # Starts the game timer
            if self.start_timer:
                self.timer = time.time()
                self.start_timer = False

            self.elapsed_time = time.time() - self.timer

        # Shuffles the tiles if button is pressed
        if self.start_shuffle:
            self.shuffle()
            self.draw_tiles()
            self.shuffle_time += 1
            if self.shuffle_time > self.shuffle_move:
                self.start_shuffle = False
                self.start_game = True
                self.start_timer = True

        # Solves the game if agent buttons is pressed
        if self.agent_activated:
            self.solve()
            self.draw_tiles()

        self.all_sprites.update()

    def run(self):
        """
            Starts the game loop.
        """
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
