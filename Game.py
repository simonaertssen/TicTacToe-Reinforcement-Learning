import numpy as np
import random
import pickle
import tqdm

def didPlayerWinAccordingToTheRules(board, player_symbol):
    score = 3 * player_symbol
    if any(board.sum(0) == score) or any(board.sum(1) == score):
        return True
    if board.diagonal().sum() == score or np.flip(board,1).diagonal().sum() == score:
        return True
    return False

class TicTacToe:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.symbols = [-1, 1]
        self.initiatePlayers()
        self.activeplayer = None
        self.activeplayerindex = 0

        self._rows = self._cols = 3
        self.board = np.zeros((self._rows, self._cols))
        self._max_moves = self._rows * self._cols
        self._currentmove = 0

        self._reward_win  = 1.0
        self._reward_lose = -1.0
        self._reward_draw = 0.2
        self._reward_draw_opponent = 0.5

        self._gameStarted = False
        self._gameEnded = False
        self._verbose = True

        self.trainresults = []
        self.testresults = []
        self.start()

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

    def start(self, startplayerindex=None):
        self.board = np.zeros((self._rows, self._cols))
        if startplayerindex is not None:
            self.activeplayerindex = startplayerindex
        else:
            for i, player in enumerate(self.players):
                if player.isHuman():
                    self.activeplayerindex = i
                    break
        self._gameStarted = True
        self.activeplayer = self.players[self.activeplayerindex]
        # print("player", self.activeplayer, "ishuman?", self.activeplayer.isHuman())
        # if not self.activeplayer.isHuman():
        #     print("playing first move")
        #     self.queryNonHumanPlayer()

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
            self.activeplayer.reward(self._reward_win)
            # message = "Player {} won the game.".format(self.activeplayer.name)
            message = self.activeplayer.symbol
            self.stop(message)

        elif self._currentmove + 1 >= self._max_moves:
            self.activeplayer.reward(self._reward_draw)
            self.players[(self.activeplayerindex+1)%2].reward(self._reward_draw_opponent)
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
        self.reset()
        self.signalGUItoReset(message_to_gui)

    def reset(self):
        self.board = np.zeros((self._rows, self._cols))
        self._gameStarted = False
        self._gameEnded = False
        self._currentmove = 0
        for player in self.players:
            player.reset()
        self.switchActivePlayer()

    def loadNewPlayer(self, new_player, index=1):
        self.players[index] = new_player
        self.initiatePlayers()

    def play(self):
        if not self.activeplayer.isHuman():
            self.queryNonHumanPlayer()
            self.reset()

    def getResults(self, result):
        self.battleresults.append(result)

    def train(self, number_of_rounds=100, number_of_battles=100):
        self._verbose = True
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

        #print(self.player1wins, self.player2wins, self.draws)
        mostwins = 0
        totalwins = 0
        bestplayer = None
        for player in self.players:
            if player.wins >= mostwins:
                mostwins = player.wins
                bestplayer = player
                totalwins += player.wins / number_of_rounds / number_of_battles
            print("{}: {} games won.".format(player.name, player.wins / number_of_rounds / number_of_battles))
        print("{} games ended in draw.".format(number_of_rounds*number_of_battles - totalwins))
        bestplayer.savePolicy()
        return gamecount, player1wins, player2wins, draws

    def test(self, number_of_rounds=100):
        for player in self.players:
            player._trainable = False
        for round in tqdm.tqdm(range(number_of_rounds)):
            self.play()
        for player in self.players:
            print("{}: {} games won.".format(player.name, player.wins / number_of_rounds))
        print("{} games ended in draw.".format((number_of_rounds*number_of_battles - self.players[0].wins - self.players[1].wins)/number_of_rounds))
