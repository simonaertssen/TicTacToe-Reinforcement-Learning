import matplotlib.pyplot as plt
from Players import RandomPlayer, MiniMaxPlayer, AlphaBetaPlayer, QPlayer, NeuralPlayer
from Game import TicTacToe

from skopt import gp_minimize

import sys
import numpy as np

def objective(args):
    lr, decay, exploration = args
    p1 = QPlayer(lr, decay, exploration)
    p2 = RandomPlayer()

    game = TicTacToe(p1, p2)
    count, p1wins, p2wins, draws = game.train(100, 100)
    score = sum(p1wins) + sum(draws) - sum(p2wins)
    return -score

res = gp_minimize(objective, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], acq_func="LCB", n_calls=15, n_random_starts=5, noise=0.1**2)
print("lr={}, decay={}, exploration={}, f()={}".format(res.x[0], res.x[1], res.x[2], res.fun))

p1 = QPlayer(lr=0.4905593711696319, decay=0.39505136184649137, exploration=0.0)
p2 = RandomPlayer()

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

p1 = QPlayer(lr=0.9, decay=0.95, exploration=0.1)
p2 = RandomPlayer()

game = TicTacToe(p1, p2)
count, p1wins, p2wins, draws = game.train(100, 100)

if p1.wins > p2.wins:
    bestQPlayer = p1
    print("Player 1 was the best player")
else:
    bestQPlayer = p2
    print("Player 2 was the best player")
bestQPlayer.savePolicy()

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

game = TicTacToe(bestQPlayer, p3)
count, p1wins, p2wins, draws = game.test(50, 5)

plt.figure()
plt.ylabel('Game outcomes in %')
plt.xlabel('Game number')

plt.plot(count, p1wins, 'g-', label='{} wins'.format(bestQPlayer.name))
plt.plot(count, p2wins, 'b-', label='{} wins'.format(p3.name))
plt.plot(count, draws, 'r-', label='Draw')

plt.grid(True)
plt.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)
plt.show()
