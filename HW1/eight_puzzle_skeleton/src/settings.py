"""
    #######################################################
    DO NOT CHANGE ANYTHING OTHER THAN GAMESIZE AND SHUFFLE.
    #######################################################
"""

###---COLORS---###
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGRAY = (40, 40, 40)
LIGHTGRAY = (100, 100, 100)
NUMBERCOLOR = (255, 0, 0)
BGCOLOUR = DARKGRAY

###---GAME SETTINGS---###
WIDTH = 1024
HEIGHT = 768
FPS = 60
TITLE = "Tile Puzzle Game"
BOARDSIZE = 384
GAMESIZE = 3
TILESIZE = BOARDSIZE // GAMESIZE
SHUFFLE = 10

###---STARTING POSITION---###
START = [(WIDTH - BOARDSIZE) // 2, (HEIGHT - BOARDSIZE) // 4]