import matplotlib.pyplot as plt
from Players import RandomPlayer, MiniMaxPlayer, AlphaBetaPlayer, ValuePlayer, NeuralPlayer
from Game import TicTacToe

import sys
import numpy as np

p1 = ValuePlayer()
p2 = ValuePlayer()

game = TicTacToe(p1, p2)
count, p1wins, p2wins, draws = game.train(100, 100)

plt.figure()
plt.ylabel('Game outcomes in %')
plt.xlabel('Game number')

plt.plot(count, p1wins, 'g-', label='{} wins'.format(p1.name))
plt.plot(count, p2wins, 'b-', label='{} wins'.format(p2.name))
plt.plot(count, draws, 'r-', label='Draw')

plt.grid(True)
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
plt.show()

sys.exit()

p1 = ValuePlayer()
p2 = ValuePlayer()

game = TicTacToe(p1, p2)
count, p1wins, p2wins, draws = game.train(100, 100)

if p1.wins > p2.wins:
    bestvalueplayer = p1
    print("Player 1 was the best player")
else:
    bestvalueplayer = p2
    print("Player 2 was the best player")
bestvalueplayer.savePolicy()


plt.figure()
plt.ylabel('Game outcomes in %')
plt.xlabel('Game number')

plt.plot(count, p1wins, 'g-', label='{} wins'.format(p1.name))
plt.plot(count, p2wins, 'b-', label='{} wins'.format(p2.name))
plt.plot(count, draws, 'r-', label='Draw')

plt.grid(True)
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
plt.draw()


p3 = AlphaBetaPlayer()

game = TicTacToe(bestvalueplayer, p3)
count, p1wins, p2wins, draws = game.test(10, 5)

plt.figure()
plt.ylabel('Game outcomes in %')
plt.xlabel('Game number')

plt.plot(count, p1wins, 'g-', label='{} wins'.format(bestvalueplayer.name))
plt.plot(count, p2wins, 'b-', label='{} wins'.format(p3.name))
plt.plot(count, draws, 'r-', label='Draw')

plt.grid(True)
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
plt.show()
