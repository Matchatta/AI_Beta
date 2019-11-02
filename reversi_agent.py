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



class EvaluateAgent(ReversiAgent):
    def __init__(self, color):
        super().__init__(color)
        self.next_move = []
        self.Score = float("-inf")

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            time.sleep(0.5)
            i = 0
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
        elif depth > 3 or actions.size==0:
            return utility
        v = float("-inf")
        for a in actions:
            new_board, new_turn = _ENV.get_next_state((board, turn), a)
            new_actions = self.GetValidAction(new_board, new_turn)
            min_result = self.MinValue(new_board, Alpha, Beta, new_actions, new_turn, depth + 1)
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
        elif depth > 3 or actions.size==0:
            return utility
        v = float("inf")
        for a in actions:
            new_board, new_turn = _ENV.get_next_state((board, turn), a)
            new_actions = self.GetValidAction(new_board, new_turn)
            max_result = self.MaxValue(new_board, Alpha, Beta, new_actions, new_turn, depth + 1)
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
        score = 0
        for d in disc:
            if board[d[0]][d[1]] == self.player:
                score += 1
        score = score / disc.size
        score *= 100
        return score

    def GetValidAction(self, board, turn):
        valids = _ENV.get_valid((board, turn))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids



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
