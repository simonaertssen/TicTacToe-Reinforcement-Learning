import numpy as np
import random
import pickle
import tqdm

def didPlayerWinAccordingToTheRules(board, player_symbol):
    score = 3 * player_symbol
    if any(board.sum(0) == score) or any(board.sum(1) == score):
        return True
    if board.diagonal().sum() == score or (board[2,0] + board[1,1] + board[0,2]) == score:
        return True
    return False

class TicTacToe:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.symbols = [-1, 1]
        self.initiatePlayers()
        self.activeplayer = None
        self.activeplayerindex = 0
        self._startplayer = 0

        self._rows = self._cols = 3
        self.board = np.zeros((self._rows, self._cols)).astype(np.int8)
        self._max_moves = self._rows * self._cols
        self._currentmove = 0

        self._gameStarted = False
        self._gameEnded = False
        self._verbose = True

        self.trainresults = []
        self.testresults = []

    def initiatePlayers(self):
        for player, symbol in zip(self.players, self.symbols):
            player.reset()
            player.symbol = symbol
            player.setSignalMovePlayed(self.processPlayerMove)

    def gameIsBusy(self):
        if self._gameStarted and not self._gameEnded:
            return True
        else:
            return False

    def returnMoveToGUI(self, move, playerindex):
        pass

    def signalGUItoReset(self, message):
        if self._verbose:
            print(message)

    def play(self):
        self._gameStarted = True
        self.activeplayerindex = self._startplayer
        self.activeplayer = self.players[self.activeplayerindex]
        if not self.activeplayer.isHuman():
            self.queryNonHumanPlayer()

    def switchActivePlayer(self):
        self.activeplayerindex = (self.activeplayerindex + 1) % 2
        self.activeplayer = self.players[self.activeplayerindex]

    def getAllAvailableBoardPositions(self):
        return [(x,y) for x,y in np.argwhere(self.board == 0)]

    def didTheCurrentPlayerWin(self):
        return didPlayerWinAccordingToTheRules(self.board, self.activeplayer.symbol)

    def humanPlayerMoved(self, move):
        if self.activeplayer.isHuman():
            self.activeplayer.playMove(move, self.board)
        else:
            raise ValueError

    def queryNonHumanPlayer(self):
        if not self.activeplayer.isHuman():
            actions = self.getAllAvailableBoardPositions()
            self.board = self.activeplayer.playMove(actions, self.board)
        else:
            raise ValueError

    def processPlayerMove(self, move):
        self.returnMoveToGUI(move, self.activeplayerindex)
        if self.didTheCurrentPlayerWin():
            self.activeplayer.wins += 1
            self.activeplayer.reward("WINNER")
            self.players[(self.activeplayerindex+1)%2].reward("LOSER")
            # message = "Player {} won the game.".format(self.activeplayer.name)
            message = self.activeplayer.symbol
            self.stop(message)

        elif self._currentmove + 1 >= self._max_moves:
            self.activeplayer.reward("CHOSEDRAW")
            self.players[(self.activeplayerindex+1)%2].reward("MADEDRAW")
            # message = "Game ended in draw."
            message = 0
            self.stop(message)

        else:
            self.switchActivePlayer()
            self._currentmove +=1
            if not self.activeplayer.isHuman():
                self.queryNonHumanPlayer()

    def stop(self, message_to_gui):
        self._gameEnded = True
        self._startplayer = (self._startplayer + 1) % 2
        self.reset()
        self.signalGUItoReset(message_to_gui)

    def reset(self):
        # Filling the board with zeros is safer than making a new instance of an np array.
        self.board.fill(0)
        self.board.astype(np.int8)
        self._gameStarted = False
        self._gameEnded = False
        self._currentmove = 0
        for player in self.players:
            player.reset()
        self.switchActivePlayer()

    def loadNewPlayer(self, new_player, index=1):
        self.players[index] = new_player
        self.initiatePlayers()

    def getResults(self, result):
        self.battleresults.append(result)

    def train(self, number_of_rounds=100, number_of_battles=100):
        self._verbose = False
        self.battleresults = []
        self.signalGUItoReset = self.getResults

        gamecount, player1wins, player2wins, draws = [], [], [], []
        for round in tqdm.tqdm(range(number_of_rounds)):
            for battle in range(number_of_battles):
                self.play()
            gamecount.append(round*number_of_battles)
            player1wins.append(self.battleresults.count(self.players[0].symbol)*100.0/number_of_battles)
            player2wins.append(self.battleresults.count(self.players[1].symbol)*100.0/number_of_battles)
            draws.append(self.battleresults.count(0)*100.0/number_of_battles)
            gamecount.append((round+1)*number_of_battles)
            player1wins.append(player1wins[-1])
            player2wins.append(player2wins[-1])
            draws.append(draws[-1])
            self.battleresults = []

        return gamecount, player1wins, player2wins, draws

    def test(self, number_of_rounds=100, number_of_battles=100):
        self._verbose = False
        for player in self.players:
            player._trainable = False
        return self.train(number_of_rounds, number_of_battles)
