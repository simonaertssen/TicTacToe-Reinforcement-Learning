import numpy as np
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
        self.trainable = True
        self._verbose = True

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

    def loadPolicy(self):
        pass

    def savePolicy(self):
        pass


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

        self._lr = 0.2
        self._decay = 0.9
        self._explore = 0.3
        self._playedmoves = []
        self._boardstates = []
        self._boardvalues = []

    def getBoardState(self, board):
        return str(board.flatten().tolist())

    def chooseMove(self, possible_moves, board):
        if np.random.rand(1) <= self._explore and self.trainable:
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
        if not self.trainable:
            return
        for move in reversed(self._playedmoves):
            boardstateindex = self._boardstates.index(move)
            value = self._boardvalues[boardstateindex]
            value += self._lr * (self._decay * prize - value)
            prize = value
            self._boardvalues[boardstateindex] = value

    def reset(self):
        self._playedmoves = []

    def savePolicy(self):
        with open("trainedPlayer.pickle", "wb") as f:
            pickle.dump((self._boardstates, self._boardvalues), f)

    def openPolicy(self):
        with open("trainedPlayer.pickle", "rb") as f:
            self._boardstates, self._boardvalues = pickle.load(f)


#class QPlayer(BasicPlayer):
