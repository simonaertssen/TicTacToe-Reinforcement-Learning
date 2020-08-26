import matplotlib.pyplot as plt
from Players import RandomPlayer, AlphaBetaPlayer, ValuePlayer, NeuralPlayer
from Game import TicTacToe

p1 = AlphaBetaPlayer("Test1")
p2 = ValuePlayer("Test2")

game = TicTacToe(p1, p2)
count, p1wins, p2wins, draws = game.train(50, 5)

plt.ylabel('Game outcomes in %')
plt.xlabel('Game number')

plt.plot(count, p1wins, 'g-', label='Player 1 wins')
plt.plot(count, p2wins, 'b-', label='Player 2 wins')
plt.plot(count, draws, 'r-', label='Draw')

plt.grid(True)
plt.legend(loc='best', shadow=True, fancybox=True, framealpha =0.7)
plt.show()
