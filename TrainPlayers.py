import matplotlib.pyplot as plt
from Players import RandomPlayer, MiniMaxPlayer, AlphaBetaPlayer, QPlayer, NeuralPlayer
from Game import TicTacToe

import sys
import numpy as np

def train(player1, player2, rounds=10, battles=10):
    game = TicTacToe(player1, player2)
    count, player1wins, player2wins, draws = game.train(rounds, battles)

    plt.figure()
    plt.ylabel('Game outcomes in %')
    plt.xlabel('Game number')

    plt.plot(count, player1wins, 'g-', label='{} wins'.format(player1.name))
    plt.plot(count, player2wins, 'b-', label='{} wins'.format(player2.name))
    plt.plot(count, draws, 'r-', label='Draw')

    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.12), shadow=False, fancybox=True, framealpha=0.3, ncol=3)
    plt.show()

# Working QPlayer versus RandomPlayer
p1 = NeuralPlayer(lr=0.1)
p2 = QPlayer(lr=0.9, lrdecay=0.9, exploration=1.0, explorationdecay=0.9)

train(p1, p2, rounds=10, battles=10)

sys.exit()

# Non-working QPlayer versus QPlayer
p1 = QPlayer(lr=0.9, lrdecay=0.9, exploration=1.0, explorationdecay=0.9)
p2 = QPlayer(lr=0.9, lrdecay=0.9, exploration=1.0, explorationdecay=0.9)

train(p1, p2, rounds=100, battles=100)

sys.exit()


# Working QPlayer versus RandomPlayer
p1 = QPlayer(lr=0.9, lrdecay=0.95, exploration=0.1, explorationdecay=0.99)
p2 = RandomPlayer()

train(p1, p2, rounds=100, battles=100)

sys.exit()


# Working QPlayer versus AlphaBetaPlayer
p1 = QPlayer(lr=0.7, lrdecay=0.95, exploration=0.3, explorationdecay=0.95)
p2 = AlphaBetaPlayer()

train(p1, p2, rounds=50, battles=50)

p1.savePolicy()
sys.exit()


# Bayesian optimisation of parameters
from skopt import gp_minimize
def objective(args):
    lr, lrdecay, exploration, explorationdecay = args
    p1 = QPlayer(lr, lrdecay, exploration, explorationdecay)
    p2 = AlphaBetaPlayer()

    game = TicTacToe(p1, p2)
    count, p1wins, p2wins, draws = game.train(50, 5)
    score = sum(p1wins) + sum(draws) - sum(p2wins)
    return -score

res = gp_minimize(objective, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], acq_func="LCB", n_calls=10, n_random_starts=5, noise=0.1**2)
print("lr={}, decay={}, exploration={}, decay={}, f()={}".format(res.x[0], res.x[1], res.x[2], res.x[3], res.fun))
