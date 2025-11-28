from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:

        opponent = 2 if self.player_id == 1 else 1

        def terminal_or_depth(b: Board, depth: int) -> bool:
            #Check if we have reached a terminal state (win,loss,draw) or if depth limit is reached
            state = b.get_board_state()
            w = self.heuristic.winning(state, self.game_n)
            if w != 0 or depth == 0:
                return True
            #If there are no valid moves left it is also terminal
            return not any(b.is_valid(c) for c in range(b.width))

        def score(b: Board) -> int:
            #evaluate the board from our perspective using the heuristic
            return self.heuristic.evaluate_board(self.player_id, b)

        def rec(b: Board, depth: int, maximizing: bool) -> int:
            #Recursive minimax function 
            if terminal_or_depth(b, depth):
                return score(b)
            if maximizing: #Our turn, try to maximize the score
                best = -10**9
                for c in range(b.width):
                    if b.is_valid(c):
                        best = max(best, rec(b.get_new_board(c, self.player_id), depth - 1, False))
                return best
            else: #oponennt's turn , try to minimze our score
                best = 10**9
                for c in range(b.width):
                    if b.is_valid(c):
                        best = min(best, rec(b.get_new_board(c, opponent), depth - 1, True))
                return best
        #Loop through all possible moves and pickthe one with the best minimax value
        best_val, best_col = -10**9, None
        for c in range(board.width):
            if board.is_valid(c):
                v = rec(board.get_new_board(c, self.player_id), self.depth - 1, False)
                if v > best_val:
                    best_val, best_col = v, c
        return best_col if best_col is not None else -1 #Return the column we want to play

    

class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        #Decide on the best move to play using minimax with alpha-beta pruning 
        opponent = 2 if self.player_id == 1 else 1

        def terminal_or_depth(b: Board, depth: int) -> bool:
            #this helps stopping if we reached a win/draw/loss, or if we reached max search depth 
            state = b.get_board_state()
            w = self.heuristic.winning(state, self.game_n)
            if w != 0 or depth == 0:
                return True
            #also stop if no valid moves are left
            return not any(b.is_valid(c) for c in range(b.width))

        def score(b: Board) -> int:
            #Evaluates the board from our perspective with the heuristic
            return self.heuristic.evaluate_board(self.player_id, b)

        def rec(b: Board, depth: int, alpha: float, beta: float, maximizing: bool) -> int:
            if terminal_or_depth(b, depth):
                return score(b)

            if maximizing:  # our turn
                value = -10**9
                for c in range(b.width):
                    if b.is_valid(c):
                        value = max(value, rec(b.get_new_board(c, self.player_id), depth - 1, alpha, beta, False))
                        alpha = max(alpha, value)
                        if alpha >= beta:  # beta cut-off
                            break
                return value
            else:  # opponent turn
                value = 10**9
                for c in range(b.width):
                    if b.is_valid(c):
                        #simulate placing our piece in column c
                        value = min(value, rec(b.get_new_board(c, opponent), depth - 1, alpha, beta, True))
                        beta = min(beta, value)
                        if alpha >= beta:  # alpha cut-off
                            break #prune the rest of this branch 
                return value
        #after recursion, pick the move that gave us the best value 
        best_val, best_col = -10**9, 0
        alpha, beta = -10**9, 10**9
        for c in range(board.width):
            if board.is_valid(c):
                v = rec(board.get_new_board(c, self.player_id), self.depth - 1, alpha, beta, False)
                if v > best_val:
                    best_val, best_col = v, c
                alpha = max(alpha, best_val)
        return best_col #return the chosen column 



class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        