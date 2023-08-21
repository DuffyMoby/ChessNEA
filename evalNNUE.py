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
    def __init__(self, w_weights: torch.tensor, w_biases: torch.tensor, 
                 b_weights: torch.tensor, b_biases: torch.tensor, 
                 acc: torch.tensor, w_king_sq=0, b_king_sq=0):
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
        # Since the only move that adds two bits is castling, which is handled by refresh()
        # we do not need a for loop when adding
        self.acc[1] += self.w_weights[:, *w_add_feature_indicies]
        # using * since indices are in list 
        self.acc[0] += self.b_weights[:, *b_add_feature_indicies] 


    def refresh(self, board: chess.Board):
        # Refresh is only called when a king moves 
        # So we only need to update king position here
        self.w_king_sq = board.king(1)
        self.b_king_sq = board.king(0)
        w_indices, b_indices = self.get_indices(board)
        # since weights are stored as shape (out_features, in_features)
        # we sum the columns since each column represents an index of the input features
        w_accumulator = self.w_biases + np.sum(self.w_weights[:, w_indices], axis=1) # axis 1 is column-wise
        b_accumulator = self.b_biases + np.sum(self.b_weights[:, b_indices], axis=1)
        self.acc[1] = w_accumulator
        self.acc[0] = b_accumulator
        
        
    def new_instance(self):
        new_acc = self.acc.copy()
        return Accumulator(self.w_weights, self.w_biases, self.b_weights, self.b_biases, new_acc, self.w_king_sq, self.b_king_sq)
    
    def get_indices(self, board: chess.Board):
        piece_bitboards = np.array([board.pawns, board.knights, board.bishops, board.rooks, board.queens], dtype=np.uint64)
        piece_bitboards = piece_bitboards[:, None] & np.array([board.occupied_co[1], board.occupied_co[0]], dtype=np.uint64)
        piece_bitboards = piece_bitboards.flatten()
        piece_bitboards = (piece_bitboards[:, None] & (1 << RANGE)) > 0

        # Multiply each row by its row number, since the piece bitboards are already in the right order
        # Bitboards represent occupancy for each square so we can just add [0 - 63] elementwise to the row
        # since we converted to a binary array with 0 as the first column or with columns arranged like so
        # -> 0, 2, 4, 8 ... 2^64
        piece_indices = (piece_bitboards * P_VALS[:, None] * 64) + RANGE

        # Then convert to halfKP indices
        w_indices = piece_indices + (self.w_king_sq) * 64 * 10
        b_indices = piece_indices + (self.b_king_sq) * 64 * 10

        # Since the bitboards also included a whole bunch on 0s
        # Multiply elementwise with original bitboard such that all indices that shouldn't exist become 0
        w_indices = w_indices * piece_bitboards
        b_indices = b_indices * piece_bitboards

        return w_indices[w_indices != 0].astype(int), b_indices[b_indices != 0].astype(int) #return indices with all 0s removed. 
    
    def get_indices_from_move(self, board: chess.Board, move: chess.Move):
        w_remove_feature_indicies = []
        w_add_feature_indicies = []
        b_remove_feature_indicies = []
        b_add_feature_indicies = []
        stm = board.turn # Side to move
        # Get halfKP index
        get_index = lambda king_sq, piece_symbol, piece_sq : piece_sq + (PIECE_TABLE[piece_symbol] + king_sq * 10) * 64

        from_sq = move.from_square
        to_sq = move.to_square
        # Piece type of piece moving 
        piece_type = board.piece_type_at(from_sq)
        piece_type = PIECE_TYPE_SYMBOL[piece_type + 6 * stm]

        # Since the piece is moving, we can remove 
        # the index of its previous square
        w_remove_feature_indicies.append(get_index(self.w_king_sq, piece_type, from_sq))
        b_remove_feature_indicies.append(get_index(self.b_king_sq, piece_type, from_sq))
        if board.is_capture(move):
            if board.is_en_passant(move):
                # The en_passant pawn will always be 1 row behind if a white pawn is moving
                # and 1 row ahead if black to move. 
                cap_sq = to_sq + (-1 * stm + (1 - stm)) * 8             
            else:
                cap_sq = to_sq

            capture_piece_type = board.piece_type_at(cap_sq)
            # Captured pieces will always be on opposing side
            capture_piece_type = PIECE_TYPE_SYMBOL[capture_piece_type + 6 * (1 - stm)] 
            # Remove the captured piece
            w_remove_feature_indicies.append(get_index(self.w_king_sq, capture_piece_type, cap_sq))
            b_remove_feature_indicies.append(get_index(self.b_king_sq, capture_piece_type, cap_sq))
            # Replace with moving piece 
            w_add_feature_indicies.append(get_index(self.w_king_sq, piece_type, to_sq))
            b_add_feature_indicies.append(get_index(self.b_king_sq, piece_type, to_sq))

        elif promotion_piece_type := move.promotion: # check for promotion 
            promotion_piece_type = PIECE_TYPE_SYMBOL[promotion_piece_type + 6 * stm]
            w_add_feature_indicies.append(get_index(self.w_king_sq, promotion_piece_type, to_sq))
            b_add_feature_indicies.append(get_index(self.b_king_sq, promotion_piece_type, to_sq))

        else:
            w_add_feature_indicies.append(get_index(self.w_king_sq, piece_type, to_sq))
            b_add_feature_indicies.append(get_index(self.b_king_sq, piece_type, to_sq))

        return w_remove_feature_indicies, w_add_feature_indicies, b_remove_feature_indicies, b_add_feature_indicies
            

        
