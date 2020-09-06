import numpy as np
import os
import torch
import random
import pickle
import tqdm

from Game import didPlayerWinAccordingToTheRules

class BasicPlayer:
    def __init__(self, symbol=None):
        self.name = self.__class__.__name__
        self.symbol = symbol
        self._human = False
        self.wins = 0
        self._trainable = False
        self._verbose = True
        self._savepath = "trained{}.pickle".format(self.name)

    def __repr__(self):
        return "{}: {}".format(self.name, self.symbol)

    def isHuman(self):
        return self._human

    def playMove(self, possible_moves, board):
        move = self.chooseMove(possible_moves, board)
        board[move] = self.symbol
        self.signalMovePlayed(move)
        return board

    def getBoardState(self, board):
        return str(board.flatten().tolist())

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

    def loadPolicy(self):
        if self._trainable:
            with open("trained{}.pickle".format(self.__class__.__name__), "rb") as f:
                self.setWeights(pickle.load(f))


class RandomPlayer(BasicPlayer):
    def __init__(self):
        super(RandomPlayer, self).__init__()

    def chooseMove(self, possible_moves, board):
        return random.choice(possible_moves)


class HumanPlayer(BasicPlayer):
    def __init__(self):
        super(HumanPlayer, self).__init__()
        self._human = True

    def playMove(self, move, board):
        board[move] = self.symbol
        self.signalMovePlayed(move)
        return board

    def chooseMove(self, move, board):
        raise NotImplementedError


class MiniMaxPlayer(BasicPlayer):
    def __init__(self):
        super(MiniMaxPlayer, self).__init__()
        self._maxdepth = 100

    def maximising(self, symbol):
        return symbol == self.symbol

    def chooseMove(self, possible_moves, board):
        best_score_index = self.rateMovesThroughMiniMax(possible_moves, board, self.symbol)
        return possible_moves[best_score_index]

    def switchPlayerSymbol(self, symbol):
        return 1 if symbol == -1 else -1

    def rateMovesThroughMiniMax(self, moves, board, playersymbol, depth=0):
        if depth > self._maxdepth:
            return [0]
        alternative_moves = moves.copy()
        alternative_board = board.copy()

        ismaximising = self.maximising(playersymbol)

        scores = []
        for move in moves:
            alternative_moves.remove(move)
            alternative_board[move] = playersymbol

            if didPlayerWinAccordingToTheRules(alternative_board, playersymbol):
                score = 1 if ismaximising else -1
            elif not (alternative_board == 0).any():
                score = 0
            else:
                score = self.rateMovesThroughMiniMax(alternative_moves, alternative_board, self.switchPlayerSymbol(playersymbol), depth + 1)
            scores.append(score)
            alternative_moves.append(move)
            alternative_board[move] = 0

        decisivescore = max(scores) if ismaximising else min(scores)
        if depth == 0:
            print("scores =", scores, "bestscoreindex =", scores.index(decisivescore))
            return scores.index(decisivescore)
        else:
            return decisivescore


class AlphaBetaPlayer(MiniMaxPlayer):
    def __init__(self):
        super(AlphaBetaPlayer, self).__init__()

    def rateMovesThroughMiniMax(self, moves, board, playersymbol, depth=0, alpha=-np.Inf, beta=np.Inf):
        if depth > self._maxdepth:
            return [0]
        printing = False
        if len(moves) < 3 and depth < 3:
            printing = True

        alternative_moves = moves.copy()
        alternative_board = board.copy()

        ismaximising = self.maximising(playersymbol)
        bestscore = - np.Inf if ismaximising else np.Inf
        bestscoreindex = 0

        for index, move in enumerate(moves):
            alternative_moves.remove(move)
            alternative_board[move] = playersymbol

            if didPlayerWinAccordingToTheRules(alternative_board, playersymbol):
                score = 1 if ismaximising else -1
            elif not (alternative_board == 0).any():
                score = 0
            else:
                score = self.rateMovesThroughMiniMax(alternative_moves, alternative_board, self.switchPlayerSymbol(playersymbol), depth + 1, alpha, beta)

            if ismaximising:
                if score > bestscore:
                    bestscore = score
                    bestscoreindex = index
                if bestscore >= beta:
                    return bestscore
                alpha = max(bestscore, alpha)
            else:
                if score < bestscore:
                    bestscore = score
                    bestscoreindex = index
                if bestscore <= alpha:
                    return bestscore
                beta = min(bestscore, beta)

            alternative_moves.append(move)
            alternative_board[move] = 0

        if depth == 0:
            return bestscoreindex
        else:
            return bestscore


class QPlayer(BasicPlayer):
    def __init__(self, lr=0.9, lrdecay=0.95, exploration=0.3, explorationdecay=1.0):
        super(QPlayer, self).__init__()
        self.wins = 0

        self._lr = lr
        self._lrdecay = lrdecay
        self._explore = exploration
        self._exploredecay = explorationdecay
        self._playedmoves = []
        self._boardpolicy = {}
        self._trainable = True

    def chooseMove(self, possible_moves, board):
        if self._trainable and np.random.rand(1) <= self._explore:
            move = random.choice(possible_moves)
        else:
            value_of_action_max = - np.Inf
            alternative_board = board.copy()
            for possible_move in possible_moves:
                alternative_board[possible_move] = self.symbol
                boardstate = self.getBoardState(alternative_board)

                value_of_action = self._boardpolicy.get(boardstate, 0)

                if value_of_action >= value_of_action_max:
                    value_of_action_max = value_of_action
                    move = possible_move
                    bestboardstate = boardstate
                alternative_board[possible_move] = 0

            self._playedmoves.append(bestboardstate)
            if bestboardstate not in self._boardpolicy:
                self._boardpolicy[bestboardstate] = 0.0
        return move

    def reward(self, prize):
        if not self._trainable:
            return
        for move in reversed(self._playedmoves):
            value = self._boardpolicy[move]
            value += self._lr * (self._lrdecay * prize - value)
            self._boardpolicy[move] = prize = value
        self._explore *= self._exploredecay

    def reset(self):
        self._playedmoves = []

    def getWeights(self):
        return self._boardpolicy

    def setWeights(self, weights):
        self._boardpolicy = weights


class NeuralPlayerBrain(torch.nn.Module):
    def __init__(self):
        super(NeuralPlayerBrain, self).__init__()
        self._inputdims  = [9, 9*9, 9]
        self._layers     = torch.nn.ModuleList()
        self._layercount = len(self._inputdims) - 1
        self._batchnorm  = torch.nn.BatchNorm1d(self.inputdims[0])
        self._dropout    = torch.nn.Dropout(p=0.3)
        self._dropoutidx = 1
        self._activation = torch.nn.ReLU()

        for i, layer in enumerate(range(self._layercount)):
            layer_to_add = torch.nn.Linear(self._inputdims[i], self._inputdims[i+1])
            torch.nn.init.normal_(layer_to_add.weight, mean=0, std=0.02)
            self._layers.append(layer_to_add)

    def forward(self, x):
        x = self._batchnorm(x)
        for layer_index in range(self._layercount - 1):
            if layer_index == self._dropoutidx:
                x = self._dropout(x)
            x = self._activation(self._layers[layer_index](x))
        return torch.sigmoid(self._layers[-1](x))


class NeuralPlayer(BasicPlayer):
    def __init__(self):
        super(NeuralPlayer, self).__init__()
        self._activebrain = NeuralPlayerBrain()
        self._learningbrain = NeuralPlayerBrain()
        self._lr = 0.1
        self._optimizer = torch.optim.SGD(self._learningbrain.parameters(), lr=self._lr)
        self._loss = torch.nn.MSELoss()

        self._playedmoves = []
        self._trainable = True

    def chooseMove(self, possible_moves, board):
        actionvalues = self._brain(torch.from_numpy(board).flatten())
        possible_actionvalues = [actionvalues[move].item() for move in possible_moves]
        index, max_actionvalues = max(possible_actionvalues, key=lambda x: x[1])
        move = possible_moves[index]
        self._playedmoves.append(move)
        return move

    def reward(self, prize):
        if not self._trainable:
            return

        self.backpropagate()
        for move in reversed(self._playedmoves):
            boardstateindex = self._boardstates.index(move)
            value = self._boardvalues[boardstateindex]
            value += self._lr * (self._decay * prize - value)
            prize = value
            self._boardvalues[boardstateindex] = value

    def backpropagate(net_context, position, move_index, target_value):
        self._optimizer.zero_grad()
        output = net_context.policy_net(convert_to_tensor(position))

        target = output.clone().detach()
        target[move_index] = target_value
        illegal_move_indexes = position.get_illegal_move_indexes()
        for mi in illegal_move_indexes:
            target[mi] = LOSS_VALUE

        loss = net_context.loss_function(output, target)
        loss.backward()
        net_context.optimizer.step()

    def reset(self):
        self._playedmoves = []

    def getWeights(self):
        return (self._boardstates, self._boardvalues)

    def setWeights(self, weights):
        self._boardstates, self._boardvalues = weights
