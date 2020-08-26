import matplotlib.pyplot as plt
from TicTacToe import ValuePlayer, RandomPlayer, Game, AlphaBetaPlayer

p1 = RandomPlayer("Test1")
p2 = ValuePlayer("Test2")

game = Game(p1, p2)
game.start()
results = game.train(10000)
plt.plot(results[9000:])
plt.show()
