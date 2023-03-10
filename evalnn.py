import torch
import numpy as np
import chess

class Evalnn_fn:
    def __init__(self,eval_net, transform):
        self.eval_net = eval_net
        self.transform = transform

    def __call__(self, board) :
        _input = self.transform(board)
        return self.eval_net(_input)


RANGE = np.arange(64, dtype=np.uint64)[::-1] # Array of ints from 1-64
B_BOARDS = np.array([n for n in range(1,12,2)]) # Odd indexes
W_BOARDS = np.array([n for n in range(0,11,2)]) # Even indexes
def board2bb(board: chess.Board) -> torch.tensor: 
    """
    Converts a chess.Board object into a binary array of bitboards
    """ 
    p = np.array([board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings], dtype=np.uint64)
    pcolor = p[:, None] & np.array([board.occupied_co[1], board.occupied_co[0]], dtype=np.uint64)
    pcolor = pcolor.flatten()
    bitboard = (pcolor[:, None] & (1 << RANGE)) > 0
    bitboard = bitboard.astype(np.float32)
    bitboard[W_BOARDS] = bitboard[W_BOARDS] * (board.turn + (board.turn - 1)) # Negate white bits if black to move
    bitboard[B_BOARDS] = bitboard[B_BOARDS] * (board.turn * -1 + (1 - board.turn) ) # Negate black bits if white to move
    return torch.from_numpy(bitboard.flatten())