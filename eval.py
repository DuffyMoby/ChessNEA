import chess

KING_WT = 200
QUEEN_WT  = 90
ROOK_WT = 50
KNIGHT_WT = 30
BISHOP_WT = 30
PAWN_WT = 10

def evaluate(board: chess.Board, whitetomove : bool):
    """
    Basic piece counting evalution
    """
    white_bb = board.occupied_co[1]
    black_bb = board.occupied_co[0]

    mat_score = KING_WT * ((board.kings & white_bb).bit_count() - (board.kings & black_bb).bit_count()) \
            + QUEEN_WT * ((board.queens & white_bb).bit_count() - (board.queens & black_bb).bit_count()) \
            + ROOK_WT * ((board.rooks & white_bb).bit_count() - (board.rooks & black_bb).bit_count()) \
            + KNIGHT_WT * ((board.knights & white_bb).bit_count() - (board.knights & black_bb).bit_count()) \
            + BISHOP_WT * ((board.bishops & white_bb).bit_count() - (board.bishops & black_bb).bit_count()) \
            + PAWN_WT * ((board.pawns & white_bb).bit_count() - (board.pawns & black_bb).bit_count())

    eval = mat_score
    if not whitetomove:
        eval = -eval
    return eval

