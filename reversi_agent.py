"""
This module contains agents that play reversi.

Version 3.0
"""

import abc
import random
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


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


class MyAgent(ReversiAgent):
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
            corner = [[0, 0], [0, 7], [7, 0], [7, 7]]
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
                stability_score =0
            score = (0.5 * disc_score) + (0.1 * mobility_score) + (0.3 * corner_score) + (0.1 * stability_score)
            return score
        else:
            return 0

    def successor_function(self, node, action):
        new_board, turn = _ENV.get_next_state((node.board, node.player), action)
        return new_board, turn

    def get_valid_actions(self, board, turn):
        valids = _ENV.get_valid((board, turn))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    def terminal_test(self, depth, node):
        return depth > 2 or self.get_valid_actions(node.board, node.player).size == 0
