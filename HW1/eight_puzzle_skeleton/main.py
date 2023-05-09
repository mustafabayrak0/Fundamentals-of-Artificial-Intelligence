from src.game import Game

if __name__ == "__main__":

    # Create a game object and run the game.
    game = Game()
    while True:
        game.new()
        game.run()