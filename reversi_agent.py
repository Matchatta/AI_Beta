"""
This module contains agents that play reversi.
Version 3.0
"""

import abc
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2

_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)
            p = Process(
                target=self.search,
                args=(
                    self._color, board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for
            `output_move_row.value` and `output_move_column.value`
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')

class KaiAgent(ReversiAgent):
    class node:
        def __init__(self, board, player, action=None, v=float("-inf")):
            self.player = player
            self.v = v
            self.action = action
            self.successor = None
            self.board = board

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            # while True:
            #     pass
            time.sleep(0.5)
            new_node = self.node(board, self.player, action=valid_actions)
            self.max_value(new_node)
            # randidx = random.randint(0, len(valid_actions) - 1)
            # random_action = valid_actions[randidx]
            output_move_row.value = new_node.successor[0]
            output_move_column.value = new_node.successor[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def max_value(self, node, depth=0, alpha=float("-inf"), beta=float("inf")):
        if self.terminal_test(depth, node):
            return self.utility(node)
        v = float("-inf")
        for a in node.action:
            new_board, turn = self.successor_function(node, a)
            new_node = self.node(new_board, turn, self.get_valid_actions(new_board, turn))
            v = max(v, self.min_value(new_node, depth=depth + 1, alpha=alpha, beta=beta))
            if depth == 0 and v != node.v:
                node.v = v
                node.successor = a
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, node, depth=0, alpha=float("-inf"), beta=float("inf")):
        if self.terminal_test(depth, node):
            return self.utility(node)
        v = float("inf")
        for a in node.action:
            new_board, turn = self.successor_function(node, a)
            new_node = self.node(new_board, turn, self.get_valid_actions(new_board, turn))
            v = min(v, self.max_value(new_node, depth=depth + 1, alpha=alpha, beta=beta))
            if depth == 0 and v != node.score:
                node.v = v
                node.successor = a
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def utility(self, node):
        if node.action.size != 0:
            disc = np.array(list(zip(*node.board.nonzero())))
            total_mobility = node.action.size + self.get_valid_actions(node.board, -1 * node.player).size
            # compute disc score
            disc_score = 0
            for d in disc:
                if self.player == node.board[d[0]][d[1]]:
                    disc_score += 1
                else:
                    disc_score -= 1
            mobility_score = 0
            if total_mobility != 0:
                mobility_score = self.get_valid_actions(node.board, self.player).size / total_mobility
            else:
                mobility_score = 0
            # compute corner score
            corner_score = 0
            total_corner = 0
            corner = [[0, 0], [0, len(node.board) - 1], [len(node.board) - 1, 0],
                      [len(node.board) - 1, len(node.board) - 1]]
            for c in corner:
                if node.board[c[0]][c[1]] == self.player:
                    corner_score += 50
                    total_corner += 1
                elif node.board[c[0]][c[1]] == -1 * self.player:
                    corner_score -= 50
                    total_corner += 1
            if total_corner != 0:
                corner_score = corner_score / total_corner
            else:
                corner_score = 0
            corner_score = corner_score / len(corner)
            # compute stability score
            stability_score = 0
            total_stability = 0
            if node.action.size != 0:
                sto_board = {}
                i = 0
                for a in node.action:
                    new_board, __ = self.successor_function(node, a)
                    sto_board[i] = new_board
                    i += 1
                for d in disc:
                    flag = False
                    for key in sto_board:
                        b = sto_board[key]
                        if node.board[d[0]][d[1]] == b[d[0]][d[1]]:
                            flag = True
                        else:
                            flag = False
                    if flag:
                        if node.board[d[0]][d[1]] == self.player:
                            stability_score += 1
                            total_stability += 1
                        else:
                            stability_score -= 1
                            total_stability += 1
            if total_stability != 0:
                stability_score = stability_score / total_stability
            else:
                stability_score = 0
            # compute avoid move score
            score_avoid_move = 0
            total_avoid_move = 0
            avoid_move = [[1, 1], [0, 1], [1, 0],
                          [0, len(node.board) - 2], [1, len(node.board) - 2], [1, len(node.board) - 1],
                          [len(node.board) - 2, 0], [len(node.board) - 2, 1], [len(node.board) - 1, 1],
                          [len(node.board) - 2, len(node.board) - 1], [len(node.board) - 2, len(node.board) - 2],
                          [len(node.board) - 1, len(node.board) - 2]]
            for am in avoid_move:
                if node.board[am[0]][am[1]] == self.player:
                    score_avoid_move -= 5
                    total_avoid_move += 1
                else:
                    score_avoid_move += 5
                    total_avoid_move += 1
            if total_avoid_move == 0:
                score_avoid_move = score_avoid_move / total_avoid_move
            else:
                score_avoid_move = 0
            score = (0.4 * disc_score) + (0.05 * mobility_score) + (0.3 * corner_score) + (0.05 * stability_score) + (0.2 * score_avoid_move)
            return score
        else:
            return 0

    @staticmethod
    def successor_function(node, action):
        new_board, turn = _ENV.get_next_state((node.board, node.player), action)
        return new_board, turn

    @staticmethod
    def get_valid_actions(board, turn):
        valid = _ENV.get_valid((board, turn))
        valid = np.array(list(zip(*valid.nonzero())))
        return valid

    def terminal_test(self, depth, node):
        return depth > 1 or self.get_valid_actions(node.board, node.player).size == 0

class NokAgent(ReversiAgent):
    class node:
        def __init__(self, board, player, action=None, v=float("-inf")):
            self.player = player
            self.v = v
            self.action = action
            self.successor = None
            self.board = board

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            # while True:
            #     pass
            time.sleep(0.5)
            new_node = self.node(board, self.player, action=valid_actions)
            self.max_value(new_node)
            # randidx = random.randint(0, len(valid_actions) - 1)
            # random_action = valid_actions[randidx]
            output_move_row.value = new_node.successor[0]
            output_move_column.value = new_node.successor[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def max_value(self, node, depth=0, alpha=float("-inf"), beta=float("inf")):
        if self.terminal_test(depth, node):
            return self.utility(node)
        v = float("-inf")
        for a in node.action:
            new_board, turn = self.successor_function(node, a)
            new_node = self.node(new_board, turn, self.get_valid_actions(new_board, turn))
            v = max(v, self.min_value(new_node, depth=depth + 1, alpha=alpha, beta=beta))
            if depth == 0 and v != node.v:
                node.v = v
                node.successor = a
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, node, depth=0, alpha=float("-inf"), beta=float("inf")):
        if self.terminal_test(depth, node):
            return self.utility(node)
        v = float("inf")
        for a in node.action:
            new_board, turn = self.successor_function(node, a)
            new_node = self.node(new_board, turn, self.get_valid_actions(new_board, turn))
            v = min(v, self.max_value(new_node, depth=depth + 1, alpha=alpha, beta=beta))
            if depth == 0 and v != node.score:
                node.v = v
                node.successor = a
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def utility(self, node):
        if node.action.size != 0:
            disc = np.array(list(zip(*node.board.nonzero())))
            disc_score = 0
            for d in disc:
                if self.player == node.board[d[0]][d[1]]:
                    disc_score += 1
                else:
                    disc_score -= 1
            return disc_score
        else:
            return 0

    @staticmethod
    def successor_function(node, action):
        new_board, turn = _ENV.get_next_state((node.board, node.player), action)
        return new_board, turn

    @staticmethod
    def get_valid_actions(board, turn):
        valid = _ENV.get_valid((board, turn))
        valid = np.array(list(zip(*valid.nonzero())))
        return valid

    def terminal_test(self, depth, node):
        return depth > 2 or self.get_valid_actions(node.board, node.player).size == 0

class HongAgent(ReversiAgent):
    def __init__(self, color):
        super().__init__(color)
        self.next_move = []
        self.Score = float("-inf")

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            time.sleep(0.5)
            i=0
            self.MaxValue(board, float("-inf"), float("inf"), valid_actions, color)
            # randidx = random.randint(0, len(valid_actions) - 1)
            # random_action = valid_actions[randidx]
            output_move_row.value = self.next_move[0]
            output_move_column.value = self.next_move[1]
        except Exception as e:
            print(self.next_move, self.Score)
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def MaxValue(self, board, alpha, beta, actions, turn, depth=0):
        Alpha = alpha
        Beta = beta
        utility = self.Utility(board)
        if 0 < depth and utility > 20:
            return utility
        elif depth > 3:
            return utility
        v = float("-inf")
        for a in actions:
            new_board, new_turn = _ENV.get_next_state((board, turn), a)
            new_actions = self.GetValidAction(new_board, new_turn)
            min_result = self.MinValue(new_board, Alpha, Beta, new_actions, new_turn, depth+1)
            v = max(v, min_result)
            if depth == 0 and v != self.Score:
                self.next_move = a
                self.Score = v
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def MinValue(self, board, alpha, beta, actions, turn, depth=0):
        Alpha = alpha
        Beta = beta
        utility = self.Utility(board)
        if 0 < depth and utility > 20:
            return utility
        elif depth > 3:
            return utility
        v = float("inf")
        for a in actions:
            new_board, new_turn = _ENV.get_next_state((board, turn), a)
            new_actions = self.GetValidAction(new_board, new_turn)
            max_result = self.MaxValue(new_board, Alpha, Beta, new_actions, new_turn, depth+1)
            v = min(v, max_result)
            if depth == 0 and v != self.Score:
                self.next_move = a
                self.Score = v
            if v >= beta:
                return v
            beta = min(beta, v)
        return v

    def Utility(self, board):
        disc = np.array(list(zip(*board.nonzero())))
        score =0
        for d in disc:
            if board[d[0]][d[1]] == self.player:
                score += 1
        score = score/disc.size
        score *= 100
        return score

    def GetValidAction(self, board, turn):
        valids = _ENV.get_valid((board, turn))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

class PedAgent(ReversiAgent):

    def __init__(self, color):
        super().__init__(color)
        self.next_move = []
        self.Score = float("-inf")

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            time.sleep(3)
            self.Max_value(board, float("-inf"), float("inf"), valid_actions, color)
            output_move_row.value = self.next_move[0]
            output_move_column.value = self.next_move[1]


        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def evaluate(self, board):
        Board = np.array(list(zip(*board.nonzero())))
        countA: int = 0
        countB: int = 0
        for row in Board:
            if board[row[0]][row[1]] == self._color:
                countA += 1
            else:
                countB += 1
        return countA - countB

    def Max_value(self, board, alpha, beta, actions, player, depth=0):
        Alpha = alpha
        Beta = beta
        score = self.evaluate(board)
        if 0 < depth:
            return score
        elif depth > 3:
            return score
        v = float("-inf")
        for x in actions:
            newboard, playerturn = _ENV.get_next_state((board, player), x)
            newactions = self.BestAction(newboard, playerturn)
            minresult = self.Min_value(newboard, Alpha, Beta, newactions, playerturn, depth+1)
            v = max(v, minresult)
            if depth == 0 and v != self.Score:
                self.next_move = x
                self.Score = v
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def Min_value(self, board, alpha, beta, actions, player, depth=0):
        Alpha = alpha
        Beta = beta
        score = self.evaluate(board)
        if 0 < depth:
            return score
        elif depth > 3:
            return score
        v = float("inf")
        for x in actions:
            newboard, playerturn = _ENV.get_next_state((board, player), x)
            newactions = self.BestAction(newboard, playerturn)
            maxresult = self.Max_value(newboard, Alpha, Beta, newactions, playerturn, depth+1)
            v = min(v, maxresult)
            if depth == 0 and v != self.Score:
                self.next_move = x
                self.Score = v
            if v >= beta:
                return v
            beta = min(beta, v)
        return v

    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1

    def BestAction(self, board, player: int) -> (np.array, np.array):
        valid:  np.array =_ENV.get_valid((board,self.getOpponent(player)))
        valid: np.array = np.array(list(zip(*valid.nonzero())))
        return valid

class HanAgent(ReversiAgent):

    def __init__(self, color):
        super().__init__(color)
        self.next_move = []
        self.Score = float("-inf")

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            time.sleep(3)
            self.Max_value(board, float("-inf"), float("inf"), valid_actions, color)
            output_move_row.value = self.next_move[0]
            output_move_column.value = self.next_move[1]


        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def evaluate(self, board):
        Board = np.array(list(zip(*board.nonzero())))
        countA: int = 0
        countB: int = 0
        for row in Board:
            if board[row[0]][row[1]] == self._color:
                countA += 1
            else:
                countB += 1
        return countA - countB

    def Max_value(self, board, alpha, beta, actions, player, depth=0):
        Alpha = alpha
        Beta = beta
        score = self.evaluate(board)
        if 0 < depth:
            return score
        elif depth > 3:
            return score
        v = float("-inf")
        for x in actions:
            newboard, playerturn = _ENV.get_next_state((board, player), x)
            newactions = self.BestAction(newboard, playerturn)
            minresult = self.Min_value(newboard, Alpha, Beta, newactions, playerturn, depth+1)
            v = max(v, minresult)
            if depth == 0 and v != self.Score:
                self.next_move = x
                self.Score = v
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def Min_value(self, board, alpha, beta, actions, player, depth=0):
        Alpha = alpha
        Beta = beta
        score = self.evaluate(board)
        if 0 < depth:
            return score
        elif depth > 3:
            return score
        v = float("inf")
        for x in actions:
            newboard, playerturn = _ENV.get_next_state((board, player), x)
            newactions = self.BestAction(newboard, playerturn)
            maxresult = self.Max_value(newboard, Alpha, Beta, newactions, playerturn, depth+1)
            v = min(v, maxresult)
            if depth == 0 and v != self.Score:
                self.next_move = x
                self.Score = v
            if v >= beta:
                return v
            beta = min(beta, v)
        return v

    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1

    def BestAction(self, board, player: int) -> (np.array, np.array):
        valid:  np.array =_ENV.get_valid((board,self.getOpponent(player)))
        valid: np.array = np.array(list(zip(*valid.nonzero())))
        return valid