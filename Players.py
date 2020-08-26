import numpy as np
import torch
import random
import pickle
import tqdm

from Game import didPlayerWinAccordingToTheRules

class BasicPlayer:
    def __init__(self, name=None, symbol=None):
        self.name = name
        self.symbol = symbol
        self._human = False
        self.wins = 0
        self._trainable = False
        self._verbose = True
        self._savepath = "trained{}.pickle".format(self.__class__.__name__)

    def __repr__(self):
        return "Player {}: {}".format(self.name, self.symbol)

    def isHuman(self):
        return self._human

    def playMove(self, possible_moves, board):
        move = self.chooseMove(possible_moves, board)
        board[move] = self.symbol
        self.signalMovePlayed(move)
        return board

    def chooseMove(self, possible_moves, board):
        raise NotImplementedError

    def signalMovePlayed(self, move):
        raise NotImplementedError

    def setSignalMovePlayed(self, callback):
        if callable(callback):
            self.signalMovePlayed = callback

    def reward(self, prize):
        pass

    def reset(self):
        pass

    def getWeights(self):
        raise NotImplementedError

    def setWeights(self, weights):
        raise NotImplementedError

    def savePolicy(self):
        if self._trainable:
            with open("trained{}.pickle".format(self.__class__.__name__), "wb") as f:
                pickle.dump(self.getWeights(), f)

    def openPolicy(self):
        if self._trainable:
            with open("trained{}.pickle".format(self.__class__.__name__), "rb") as f:
                self.setWeights(pickle.load(f))


class RandomPlayer(BasicPlayer):
    def __init__(self, name):
        super(RandomPlayer, self).__init__(name)

    def chooseMove(self, possible_moves, board):
        return random.choice(possible_moves)


class HumanPlayer(BasicPlayer):
    def __init__(self, name):
        super(HumanPlayer, self).__init__(name)
        self._human = True

    def playMove(self, move, board):
        board[move] = self.symbol
        self.signalMovePlayed(move)
        return board

    def chooseMove(self, move, board):
        raise NotImplementedError


class MiniMaxPlayer(BasicPlayer):
    def __init__(self, name):
        super(MiniMaxPlayer, self).__init__(name)
        self._maxdepth = 100

    def maximising(self, symbol):
        return symbol == self.symbol

    def chooseMove(self, possible_moves, board):
        best_score_index = self.rateMovesThroughMiniMax(possible_moves, board, self.symbol, 0)
        return possible_moves[best_score_index]

    def switchPlayerSymbol(self, symbol):
        return 1 if symbol == -1 else -1

    def rateMovesThroughMiniMax(self, moves, board, playersymbol, depth):
        if depth > self._maxdepth:
            return [0]
        alternative_moves = moves.copy()
        alternative_board = board.copy()

        scores = []
        for move in moves:
            alternative_moves.remove(move)
            alternative_board[move] = playersymbol
            if didPlayerWinAccordingToTheRules(alternative_board, playersymbol):
                score = 1 if self.maximising(playersymbol) else -1
            elif not (alternative_board == 0).any():
                score = 0
            else:
                score = self.rateMovesThroughMiniMax(alternative_moves, alternative_board, self.switchPlayerSymbol(playersymbol), depth + 1)
            scores.append(score)
            alternative_moves.append(move)
            alternative_board[move] = 0

        decisivescore = max(scores) if self.maximising(playersymbol) else min(scores)
        if depth == 0:
            return scores.index(decisivescore)
        else:
            return decisivescore


class AlphaBetaPlayer(MiniMaxPlayer):
    def __init__(self, name):
        super(AlphaBetaPlayer, self).__init__(name)
        self.alpha = - np.Inf
        self.beta = np.Inf

    # This function became a lot larger than expected, due to the double types of return statements (for single depth we only want the index of the best move)
    # and due to combining maximising and minimising in one statement.
    def rateMovesThroughMiniMax(self, moves, board, playersymbol, depth):
        if depth > self._maxdepth:
            return 0
        ismaximising = self.maximising(playersymbol)
        if didPlayerWinAccordingToTheRules(board, playersymbol):
            if ismaximising:
                if depth == 0:
                    return 0
                else:
                    return 1
            else:
                if depth == 0:
                    return 0
                else:
                    return -1
        elif not (board == 0).any():
            return 0

        alternative_moves = moves.copy()
        alternative_board = board.copy()

        bestscore = - np.Inf if ismaximising else np.Inf
        bestscoreindex = None

        for index, move in enumerate(moves):
            alternative_moves.remove(move)
            alternative_board[move] = playersymbol
            score = self.rateMovesThroughMiniMax(alternative_moves, alternative_board, self.switchPlayerSymbol(playersymbol), depth + 1)
            if ismaximising:
                if score > bestscore:
                    bestscore = score
                    bestscoreindex = index
                if bestscore >= self.beta:
                    if depth == 0:
                        return bestscoreindex
                    else:
                        return bestscore
                self.alpha = max(bestscore, self.alpha)
            else:
                if score < bestscore:
                    bestscore = score
                    bestscoreindex = index
                if bestscore <= self.alpha:
                    if depth == 0:
                        return bestscoreindex
                    else:
                        return bestscore
                self.beta = min(bestscore, self.beta)

            alternative_moves.append(move)
            alternative_board[move] = 0

        if depth == 0:
            return bestscoreindex
        else:
            return bestscore


class ValuePlayer(BasicPlayer):
    def __init__(self, name):
        super(ValuePlayer, self).__init__(name)
        self.wins = 0

        self._lr = 0.1
        self._decay = 0.9
        self._explore = 0.3
        self._playedmoves = []
        self._boardstates = []
        self._boardvalues = []
        self._trainable = True

    def getBoardState(self, board):
        return str(board.flatten().tolist())

    def chooseMove(self, possible_moves, board):
        if np.random.rand(1) <= self._explore and self._trainable:
            move = random.choice(possible_moves)
        else:
            value_of_action_max = - np.Inf
            bestboardstate = None
            alternative_board = board.copy()
            for possible_move in possible_moves:
                alternative_board[possible_move] = self.symbol
                boardstate = self.getBoardState(alternative_board)
                if boardstate in self._boardstates:
                    index = self._boardstates.index(boardstate)
                    value_of_action = self._boardvalues[index]
                else:
                    value_of_action = 0

                if value_of_action > value_of_action_max:
                    value_of_action_max = value_of_action
                    move = possible_move
                    bestboardstate = boardstate
                alternative_board[possible_move] = 0

            self._playedmoves.append(bestboardstate)
            if bestboardstate not in self._boardstates:
                self._boardstates.append(bestboardstate)
                valueOfMoveToBeDetermined = 0
                self._boardvalues.append(valueOfMoveToBeDetermined)
        return move

    def reward(self, prize):
        if not self._trainable:
            return
        for move in reversed(self._playedmoves):
            boardstateindex = self._boardstates.index(move)
            value = self._boardvalues[boardstateindex]
            value += self._lr * (self._decay * prize - value)
            prize = value
            self._boardvalues[boardstateindex] = value

    def reset(self):
        self._playedmoves = []

    def getWeights(self):
        return (self._boardstates, self._boardvalues)

    def setWeights(self, weights):
        self._boardstates, self._boardvalues = weights


class NeuralPlayerBrain(torch.nn.Module):
    def __init__(self):
        super(NeuralPlayerBrain, self).__init__()
        self.inputdims  = [9, 9*9, 9]
        self.layers     = torch.nn.ModuleList()
        self.layercount = len(self.inputdims) - 1
        self.batchnorm  = torch.nn.BatchNorm1d(self.inputdims[0])
        self.dropout    = torch.nn.Dropout(p=0.3)
        self.dropoutidx = 1
        self.activation = torch.nn.LeakyReLU(0.2)
        self.lr         = 0.02

        for i, layer in enumerate(range(self.layercount)):
            layer_to_add = torch.nn.Linear(self.inputdims[i], self.inputdims[i+1])
            torch.nn.init.normal_(layer_to_add.weight, mean=0, std=0.02)
            self.layers.append(layer_to_add)

    def forward(self, x):
        x = self.batchnorm(x)
        for layer_index in range(self.layercount - 1):
            if layer_index == self.dropoutidx:
                x = self.dropout(x)
            x = self.activation(self.layers[layer_index](x))
        return self.layers[-1](x)


class NeuralPlayer(BasicPlayer):
    def __init__(self, name):
        super(NeuralPlayer, self).__init__(name)
        self._brain = NeuralPlayerBrain()
        self._trainable = True

    def chooseMove(self, possible_moves, board):
        bestactionvalue = - float("Inf")
        board = torch.from_numpy(board)
        alternative_board = board.clone()
        for i, possible_move in enumerate(possible_moves):
            alternative_board[possible_move] = self.symbol
            value = self._brain(alternative_board.flatten())
            if value > bestactionvalue:
                bestactionvalue = value
                move = possible_move
            alternative_board[possible_move] = 0
        return move

    def reward(self, prize):
        if not self._trainable:
            return
        for move in reversed(self._playedmoves):
            boardstateindex = self._boardstates.index(move)
            value = self._boardvalues[boardstateindex]
            value += self._lr * (self._decay * prize - value)
            prize = value
            self._boardvalues[boardstateindex] = value

    def reset(self):
        self._playedmoves = []

    def getWeights(self):
        return (self._boardstates, self._boardvalues)

    def setWeights(self, weights):
        self._boardstates, self._boardvalues = weights
