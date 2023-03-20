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


class Accumulator:
    def __init__(self, w_weights, b_weights) -> None:
        self.w_weights = w_weights
        self.b_weights = b_weights
        self.acc_size = self.w_weights.shape[0]
        self.acc = torch.zeros((2, self.acc_size)) # tensor of size (2(stm) * out_features)
        
    def update(self, board: chess.Board, move: chess.Move):
        w_remove_feature_indicies, w_add_feature_indicies, b_remove_feature_indicies, b_add_feature_indicies = self.get_indices_from_move(board, move)
        self.acc[1] = self.acc[1].index_add(1, w_remove_feature_indicies, self.acc[1], alpha= -1)
        self.acc[1] = self.acc[1].index_add(1, w_add_feature_indicies, self.acc[1], alpha= 1)
        self.acc[0] = self.acc[0].index_add(1, b_remove_feature_indicies, self.acc[0], alpha= -1)
        self.acc[0] = self.acc[0].index_add(1, b_add_feature_indicies, self.acc[0], alpha= 1)

    def refresh(self, board: chess.Board):
        w_indices, b_indices = self.get_indices(board)
        w_accumulator = torch.sum(self.w_weights[: ,w_indices])
        # since weights are stored as shape (out_features, in_features)
        # we sum the columns since each column represents an index of the input features
        b_accumulator = torch.sum(self.b_weights[: ,b_indices])
        self.acc[1] = w_accumulator
        self.acc[0] = b_accumulator
        
    def new_instance(self):
        return Accumulator(self.weights, self.white_accumulator, self.black_accumulator)
    
    def get_indices(self, board: chess.Board):
        piece_bitboards = np.array([board.pawns, board.knights, board.bishops, board.rooks, board.queens], dtype=np.uint64)
        piece_bitboards = piece_bitboards[:, None] & np.array([board.occupied_co[1], board.occupied_co[0]], dtype=np.uint64)
        piece_bitboards = piece_bitboards.flatten()
        piece_bitboards = (piece_bitboards[:, None] & (1 << RANGE)) > 0

        piece_indices = (piece_bitboards * P_VALS[:, None] * 64) + RANGE
        w_indices = piece_indices + (board.king(1)) * 64 * 10
        w_indices = w_indices * piece_bitboards
        b_indices = piece_indices + (board.king(0)) * 64 * 10
        b_indices = b_indices * piece_bitboards
        return w_indices[w_indices != 0].astype(int), b_indices[b_indices != 0].astype(int)
    
    def get_indices_from_move(self, board: chess.Board, move: chess.Move):
        w_king_sq = board.king(1)
        b_king_sq = board.king(0)
        get_index = lambda king_sq, piece_symbol, piece_sq : piece_sq + (PIECE_TABLE[piece_symbol] + king_sq * 10) * 64
        from_sq = move.from_square
        to_sq = move.to_square
        piece_type = board.piece_at(from_sq).symbol()
        w_remove_feature_indicies = []
        w_add_feature_indicies = []
        b_remove_feature_indicies = []
        b_add_feature_indicies = []
        w_remove_feature_indicies.append(get_index(w_king_sq, piece_type, from_sq))
        b_remove_feature_indicies.append(get_index(b_king_sq, piece_type, from_sq))
        if board.is_capture(move):
            capture_piece_type = board.piece_at(to_sq).symbol()
            w_remove_feature_indicies.append(get_index(w_king_sq, capture_piece_type, to_sq))
            b_remove_feature_indicies.append(get_index(b_king_sq, capture_piece_type, to_sq))
            w_add_feature_indicies.append(get_index(w_king_sq, piece_type, to_sq))
            b_add_feature_indicies.append(get_index(b_king_sq, piece_type, to_sq))

        elif promotion_piece_type := move.promotion:
            promotion_piece_type = promotion_piece_type.symbol()
            w_add_feature_indicies.append(get_index(w_king_sq, promotion_piece_type, to_sq))
            b_add_feature_indicies.append(get_index(b_king_sq, promotion_piece_type, to_sq))

        else:
            w_add_feature_indicies.append(get_index(w_king_sq, piece_type, to_sq))
            b_add_feature_indicies.append(get_index(b_king_sq, piece_type, to_sq))

        return w_remove_feature_indicies, w_add_feature_indicies, b_remove_feature_indicies, b_add_feature_indicies
            

        
