import sys
import numpy as np
from inspect import isclass
from PyQt5 import Qt, QtGui, QtCore, QtWidgets

import Game
import Players

buttonWidth = buttonHeight = 128

class TicTacToeButton(QtWidgets.QPushButton):
    def __init__(self, position, query_gamestart, start_game, play_move):
        super(TicTacToeButton, self).__init__()
        self.setMinimumSize(QtCore.QSize(buttonWidth, buttonHeight))
        self.clicked.connect(self.onClicked)

        self.position = position
        self.didGameStart = query_gamestart
        self.startGame = start_game
        self.playMove = play_move
        self.drawSymbols = ["X", "O"]

    def mark(self, index):
        if self.isEnabled():
            self.setText(self.drawSymbols[index])
            self.setEnabled(False)

    def onClicked(self):
        if not self.isEnabled():
            return
        if not self.didGameStart():
            self.startGame()
        self.playMove(self.position)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Variables:
        self.game = Game.TicTacToe(Players.HumanPlayer("Human"), Players.ValuePlayer("Bot"))
        self.game.returnMoveToGUI = self.drawMoveOnBoard
        self.game.signalGUItoReset = self.informUserGameEnded

        # Qt window
        self.setWindowTitle("Tic Tac Toe Game")
        self.buttongrid = QtWidgets.QWidget(self)
        self.setCentralWidget(self.buttongrid)
        self.frame = QtWidgets.QFrame(self.buttongrid)
        self.frame.setGeometry(QtCore.QRect(0, 0, 3*buttonWidth, int(3.5*buttonWidth)))
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(0)
        self.grid = QtWidgets.QGridLayout(self.frame)

        # Start a new game
        self.newGameButton = QtWidgets.QPushButton("New Game")
        self.newGameButton.setMinimumSize(QtCore.QSize(buttonWidth, 8))
        self.newGameButton.clicked.connect(self.reset)
        self.grid.addWidget(self.newGameButton, 0, 0)

        # Train the players
        self.trainButton = QtWidgets.QPushButton("Train")
        self.trainButton.setMinimumSize(QtCore.QSize(buttonWidth, 8))
        self.trainButton.clicked.connect(self.train)
        self.grid.addWidget(self.trainButton, 0, 2)

        # Find all player classes:
        playerClasses = [x for x in dir(Players) if isclass(getattr(Players, x)) and 'Player' in x and 'Basic' not in x and 'Brain' not in x]
        self.choosePlayerComboBox = QtWidgets.QComboBox()
        for player in playerClasses:
            self.choosePlayerComboBox.addItem(player)
            self.choosePlayerComboBox.setMinimumSize(QtCore.QSize(buttonWidth, 8))
        self.choosePlayerComboBox.setCurrentIndex(4)
        self.choosePlayerComboBox.activated.connect(self.loadNewPlayer)
        self.grid.addWidget(self.choosePlayerComboBox, 0, 1)

        # Add game buttons
        for i in range(0,self.game._rows):
            for j in range(0, self.game._cols):
                newButton = TicTacToeButton((i,j), self.game.gameIsBusy, self.game.play, self.game.humanPlayerMoved)
                newButton.setFont(QtGui.QFont('Helvetica', 60))
                self.grid.addWidget(newButton, i+1, j)
        self.setGeometry(300, 300, 3*buttonWidth + 5, 3*buttonHeight + 60)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def drawMoveOnBoard(self, newmove, playerindex):
        self.grid.itemAt(newmove[0]*self.grid.columnCount() + newmove[1] + 3).widget().mark(playerindex)

    def informUserGameEnded(self, result):
        info = QtWidgets.QMessageBox()
        info.setWindowTitle("End of the game")
        if result == 0:
            message = "The game ended in a draw"
        elif result == -1 or result == 1:
            for player in self.game.players:
                if player.symbol == result:
                    message = "Player {} has won the game".format(player.name)
        if info.information(self,'', message, info.Yes) == info.Yes:
            self.reset()

    def reset(self):
        for i in range(self.grid.count()):
            widget = self.grid.itemAt(i).widget()
            if isinstance(widget, TicTacToeButton):
                widget.setEnabled(True)
                widget.setText("")
        self.game.play()


    def loadNewPlayer(self, boxIndex):
        self.reset()
        newPLayeName = self.choosePlayerComboBox.itemText(boxIndex)
        print("Loading player {}".format(newPLayeName))
        newPlayer = getattr(Players, newPLayeName)
        newPlayer = newPlayer(newPLayeName)
        newPlayer.loadPolicy()
        self.game.loadNewPlayer(newPlayer)

    def train(self, rounds=1000):
        self.game.play()
        self.game.train(rounds)

    def test(self):
        self.game.play()
        self.game.test()


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
