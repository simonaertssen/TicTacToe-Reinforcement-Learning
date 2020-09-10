import os
import torch
import pickle
import tqdm

import numpy as np
from numpy.random import rand

from random import choice
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
        self.rewards = {"WINNER": 1.0, "LOSER":-1.0, "MADEDRAW":0.5, "CHOSEDRAW":0.5}

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
        return board.tostring()

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
        return choice(possible_moves)


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

    def chooseMove(self, possible_moves, board):
        best_score_index = self.rateMovesThroughMiniMax(possible_moves, board, self.symbol)
        return possible_moves[best_score_index]

    def switchPlayerSymbol(self, symbol):
        return 1 if symbol == -1 else -1

    def rateMovesThroughMiniMax(self, moves, board, playersymbol, depth=0):
        if depth > self._maxdepth:
            return [0]

        ismaximising = playersymbol == self.symbol
        alternate_moves = moves.copy()
        alternate_board = board.copy()

        scores = []
        for index, move in enumerate(moves):
            alternate_moves.remove(move)
            alternate_board[move] = playersymbol

            if didPlayerWinAccordingToTheRules(alternate_board, playersymbol):
                score = 1 if ismaximising else -1
            elif not (alternate_board == 0).any():
                score = 0
            else:
                score = self.rateMovesThroughMiniMax(alternate_moves, alternate_board, self.switchPlayerSymbol(playersymbol), depth + 1)
            scores.append(score)

            alternate_moves.insert(index, move)
            alternate_board[move] = 0

        decisivescore = max(scores) if ismaximising else min(scores)
        if depth == 0:
            return scores.index(decisivescore)
        else:
            return decisivescore


class AlphaBetaPlayer(MiniMaxPlayer):
    def __init__(self):
        super(AlphaBetaPlayer, self).__init__()
        self._moves = {}

    def chooseMove(self, possible_moves, board):
        boardstate = self.getBoardState(board)
        if boardstate in self._moves:
            return self._moves[boardstate]
        else:
            best_score_index = self.rateMovesThroughMiniMax(possible_moves, board, self.symbol)
            self._moves[boardstate] = possible_moves[best_score_index]
            return possible_moves[best_score_index]

    def rateMovesThroughMiniMax(self, moves, board, playersymbol, depth=0, alpha=-np.Inf, beta=+np.Inf):
        if depth > self._maxdepth:
            return [0]

        ismaximising = playersymbol == self.symbol
        bestscore = - np.Inf if ismaximising else np.Inf
        bestscoreindex = 0

        alternate_moves = moves.copy()
        alternate_board = board.copy()

        for index, move in enumerate(moves):
            alternate_moves.remove(move)
            alternate_board[move] = playersymbol

            if didPlayerWinAccordingToTheRules(alternate_board, playersymbol):
                score = 1 if ismaximising else -1
            elif not (alternate_board == 0).any():
                score = 0
            else:
                score = self.rateMovesThroughMiniMax(alternate_moves, alternate_board, self.switchPlayerSymbol(playersymbol), depth + 1, alpha, beta)

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

            alternate_moves.append(move)
            alternate_board[move] = 0

        if depth == 0:
            return bestscoreindex
        else:
            return bestscore

        def getWeights(self):
            return self._moves

        def setWeights(self, weights):
            self._moves = weights


class QPlayer(BasicPlayer):
    def __init__(self, lr=0.5, lrdecay=0.99, exploration=0.3, explorationdecay=0.95):
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
        if self._trainable and rand(1) <= self._explore:
            move = choice(possible_moves)
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
        prize = self.rewards[prize]
        for move in reversed(self._playedmoves):
            value = self._boardpolicy[move]
            value += self._lr * (prize - value)
            self._boardpolicy[move] = prize = value
        self._explore *= self._exploredecay
        self._lr *= self._lrdecay

    def reset(self):
        self._playedmoves = []

    def getWeights(self):
        return self._boardpolicy

    def setWeights(self, weights):
        self._boardpolicy = weights


class SmartPlayerBrain(torch.nn.Module):
    def __init__(self, nodes):
        super(SmartPlayerBrain, self).__init__()
        self._inputdims  = [9, nodes, 9]
        self._layers     = torch.nn.ModuleList()
        self._layercount = len(self._inputdims) - 1
        self._dropout    = torch.nn.Dropout(p=0.3)
        self._dropoutidx = 1
        self._activation = torch.nn.ReLU()

        for i, layer in enumerate(range(self._layercount)):
            layer_to_add = torch.nn.Linear(self._inputdims[i], self._inputdims[i+1])
            #torch.nn.init.normal_(layer_to_add.weight, mean=0, std=0.02)
            self._layers.append(layer_to_add)

    def forward(self, x):
        x = x.float()
        for layer_index in range(self._layercount - 1):
            if layer_index == self._dropoutidx:
                x = self._dropout(x)
            x = self._activation(self._layers[layer_index](x))
        return torch.sigmoid(self._layers[-1](x))


class SmartPlayer(BasicPlayer):
    def __init__(self, nodes=9, lr=0.1, lrdecay=0.98, discount=0.99):
        super(SmartPlayer, self).__init__()
        self._activebrain = SmartPlayerBrain(nodes)   # = target net
        self._learningbrain = SmartPlayerBrain(nodes) # = policy net
        self._lr = lr
        self._lrdecay = lrdecay
        self._discount = discount
        self._optimizer = torch.optim.SGD(self._learningbrain.parameters(), lr=self._lr, momentum=0.9)
        #self._optimizer = torch.optim.Adam(self._learningbrain.parameters(), lr=self._lr)

        self._loss = torch.nn.MSELoss()
        self.rewards = {"WINNER": 1.0, "LOSER":-1.0, "MADEDRAW":0.5, "CHOSEDRAW":0.5}

        self._playedmoves = []
        self._trainable = True

    def chooseMove(self, possible_moves, board):
        torchboard = torch.from_numpy(board).flatten()
        actionvalues = self._activebrain(torchboard).reshape(3,3)
        possible_actionvalues = [actionvalues[move].item() for move in possible_moves]
        index, max_actionvalues = max(enumerate(possible_actionvalues), key=lambda x: x[1])
        move = possible_moves[index]
        self._playedmoves.append((board, move))
        return move

    def reward(self, prize):
        if not self._trainable:
            return
        prize = self.rewards[prize]
        for i, (board, move) in enumerate(reversed(self._playedmoves)):
            if i != 0:
                with torch.no_grad():
                    torchboard = torch.from_numpy(board).flatten()
                    actionvalues = self._activebrain(torchboard)
                    maxvalue = torch.max(actionvalues).item()
                    prize = self._discount * maxvalue
            self.backpropagate(board, move, prize)
        self._activebrain.load_state_dict(self._learningbrain.state_dict())

        self._lr *= self._lrdecay
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._lr

    def backpropagate(self, playedboard, playedmove, prize):
        self._optimizer.zero_grad()
        torchboard = torch.from_numpy(playedboard).type(torch.int8)
        actionvalues = self._learningbrain(torchboard.flatten()).reshape(3,3)
        targets = actionvalues.clone().detach()
        targets[torchboard == 0] = 0
        targets[playedmove] = prize

        loss = self._loss(actionvalues, targets)
        loss.backward()
        self._optimizer.step()

    def reset(self):
        self._playedmoves = []

    def getWeights(self):
        return self._activebrain.state_dict()

    def setWeights(self, weights):
        self._activebrain.load_state_dict(weights)
