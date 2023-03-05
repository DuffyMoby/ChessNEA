import chess
import math
from eval import evaluate


def searchAB(eval_fn, board: chess.Board, depth, alpha = -math.inf, beta = +math.inf):
    """
    Negamax search with alpha-beta

    A naive implementation of alpha beta for negamax could be:

    If white node:
        alpha = max(alpha, eval)
        if alpha > beta:
            break

    If black node:
        beta = max(beta, eval)
        if beta > alpha:
            break
    
    However, since alpha and beta represents the best and worst eval for a colour,
    then for each recursive call and increase in depth - next alpha <- -beta, next beta <- -alpha, 
    as the best move for one colour is the worst move for the opposing colour

    """
    if depth == 0 :
        return eval_fn(board), None

    maxeval = -math.inf
    bestmove = None
    # TODO move ordering, iterative deeping, tranposition table 
    for move in board.legal_moves:
        board.push(move)
        eval, _ = searchAB(eval_fn, board, depth - 1, -beta, -alpha)
        eval = -eval
        board.pop()
        if eval > maxeval:
            maxeval = eval
            bestmove = move
        
        alpha = max(alpha, maxeval)
        if alpha >= beta:
            break

    
    return maxeval, bestmove

def search(board: chess.Board, whitetomove: bool, depth):
    """
    Basic Negamax search
    """
    if depth == 0 :
        return evaluate(board, whitetomove), None

    maxeval = -math.inf
    bestmove = None
    for move in board.legal_moves:
        board.push(move)
        eval, _ = search(board, (not whitetomove), depth - 1)
        eval = -eval
        board.pop()
        if eval > maxeval:
            maxeval = eval
            bestmove = move

    return maxeval, bestmove
    
# TODO rewrite into class -> to store accumulator after implementing NNUE
# TODO figure out how to interpret move and how to refresh accumulator with board object 