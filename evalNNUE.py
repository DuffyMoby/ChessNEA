import torch
import numpy as np
import chess

RANGE = np.arange(64, dtype=np.uint64)
P_VALS = np.arange(10)
PIECE_TABLE = {
    'P' : 0,
    'p' : 1,
    'N' : 2,
    'n' : 3,
    'B' : 4,
    'b' : 5,
    'R' : 6,
    'r' : 7,
    'Q' : 8,
    'q' : 9
}
PIECE_TYPE_SYMBOL = {
    1 : 'p',
    2 : 'n',
    3 : 'b',
    4 : 'r',
    5 : 'q',
    6 : 'k',
    7 : 'P',
    8 : 'N',
    9 : 'B',
    10 : 'R',
    11 : 'Q',
    12 : 'K'
} # The value of the white pieces are 6 + the respective value of the black pieces
# so the symbol can be found easily from board.piece_type_at(square) which returns 1 if pawn, 2 if knight etc.


class Accumulator:
    def __init__(self, w_weights: torch.tensor, w_biases: torch.tensor, b_weights: torch.tensor, b_biases: torch.tensor, acc: torch.tensor, w_king_sq=0, b_king_sq=0) -> None:
        self.w_weights = w_weights
        self.w_biases = w_biases
        self.b_weights = b_weights
        self.b_biases = b_biases
        self.acc = acc
        self.w_king_sq = w_king_sq
        self.b_king_sq = b_king_sq
        
    def update(self, board: chess.Board, move: chess.Move):
        w_remove_feature_indicies, w_add_feature_indicies, b_remove_feature_indicies, b_add_feature_indicies = self.get_indices_from_move(board, move)
        for i in range(len(w_remove_feature_indicies)):
            self.acc[1] -= self.w_weights[:, w_remove_feature_indicies[i]]
            self.acc[0] -= self.b_weights[:, b_remove_feature_indicies[i]]
        self.acc[1] += self.w_weights[:, *w_add_feature_indicies] 
        # Since the only move that adds two bits is castling, which is handled by refresh()
        self.acc[0] += self.b_weights[:, *b_add_feature_indicies] 


    def refresh(self, board: chess.Board):
        self.w_king_sq = board.king(1)
        self.b_king_sq = board.king(0)
        w_indices, b_indices = self.get_indices(board, self.w_king_sq, self.b_king_sq)
        w_accumulator = self.w_biases + np.sum(self.w_weights[:, w_indices], axis=1)
        # since weights are stored as shape (out_features, in_features)
        # we sum the columns since each column represents an index of the input features
        b_accumulator = self.b_biases + np.sum(self.b_weights[:, b_indices], axis=1)
        self.acc[1] = w_accumulator
        self.acc[0] = b_accumulator
        
        
    def new_instance(self):
        new_acc = self.acc.copy()
        return Accumulator(self.w_weights, self.w_biases, self.b_weights, self.b_biases, new_acc, self.w_king_sq, self.b_king_sq)
    
    def get_indices(self, board: chess.Board, w_king_sq, b_king_sq):
        piece_bitboards = np.array([board.pawns, board.knights, board.bishops, board.rooks, board.queens], dtype=np.uint64)
        piece_bitboards = piece_bitboards[:, None] & np.array([board.occupied_co[1], board.occupied_co[0]], dtype=np.uint64)
        piece_bitboards = piece_bitboards.flatten()
        piece_bitboards = (piece_bitboards[:, None] & (1 << RANGE)) > 0

        piece_indices = (piece_bitboards * P_VALS[:, None] * 64) + RANGE
        w_indices = piece_indices + (w_king_sq) * 64 * 10
        w_indices = w_indices * piece_bitboards
        b_indices = piece_indices + (b_king_sq) * 64 * 10
        b_indices = b_indices * piece_bitboards
        return w_indices[w_indices != 0].astype(int), b_indices[b_indices != 0].astype(int)
    
    def get_indices_from_move(self, board: chess.Board, move: chess.Move):
        stm = board.turn
        get_index = lambda king_sq, piece_symbol, piece_sq : piece_sq + (PIECE_TABLE[piece_symbol] + king_sq * 10) * 64
        from_sq = move.from_square
        to_sq = move.to_square
        piece_type = board.piece_type_at(from_sq)
        piece_type = PIECE_TYPE_SYMBOL[piece_type + 6 * stm]
        w_remove_feature_indicies = []
        w_add_feature_indicies = []
        b_remove_feature_indicies = []
        b_add_feature_indicies = []
        w_remove_feature_indicies.append(get_index(self.w_king_sq, piece_type, from_sq))
        b_remove_feature_indicies.append(get_index(self.b_king_sq, piece_type, from_sq))
        if board.is_capture(move):
            if board.is_en_passant(move):
                cap_sq = to_sq + (-1 * stm + (1 - stm)) * 8

            else:
                cap_sq = to_sq

            capture_piece_type = board.piece_type_at(cap_sq)
            capture_piece_type = PIECE_TYPE_SYMBOL[capture_piece_type + 6 * (1 - stm)] # captured pieces will always be on opposing side
            w_remove_feature_indicies.append(get_index(self.w_king_sq, capture_piece_type, cap_sq))
            b_remove_feature_indicies.append(get_index(self.b_king_sq, capture_piece_type, cap_sq))
            w_add_feature_indicies.append(get_index(self.w_king_sq, piece_type, to_sq))
            b_add_feature_indicies.append(get_index(self.b_king_sq, piece_type, to_sq))

        elif promotion_piece_type := move.promotion:
            promotion_piece_type = PIECE_TYPE_SYMBOL[promotion_piece_type + 6 * stm]
            w_add_feature_indicies.append(get_index(self.w_king_sq, promotion_piece_type, to_sq))
            b_add_feature_indicies.append(get_index(self.b_king_sq, promotion_piece_type, to_sq))

        else:
            w_add_feature_indicies.append(get_index(self.w_king_sq, piece_type, to_sq))
            b_add_feature_indicies.append(get_index(self.b_king_sq, piece_type, to_sq))

        return w_remove_feature_indicies, w_add_feature_indicies, b_remove_feature_indicies, b_add_feature_indicies
            

        
