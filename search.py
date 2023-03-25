import chess
import math
from evalNNUE import Accumulator
from nn.model import halfKP_Net, nnEval_model_halfKP
import torch

class SearcherNNUE:
    """
    Wrapper for search function so accumulator can be handled internally
    """
    def __init__(self, net):
        # get weights from network to parse into accumulator
        self.w_weights = net.L_0_white[0].weight.detach()
        self.w_biases = net.L_0_white[0].bias.detach()
        self.b_weights = net.L_0_black[0].weight.detach()
        self.b_biases = net.L_0_black[0].bias.detach()
        self.net = net
        self.accumulator = Accumulator(self.w_weights, self.w_biases, self.b_weights, self.b_biases, torch.zeros((2, self.w_weights.shape[0]), requires_grad=False))


    def search(self, board, depth):
        self.accumulator.refresh(board)
        evaluation, move = self.search_helper(self.net, self.accumulator, board, depth)
        return evaluation, move


    def search_helper(self, net: halfKP_Net, accumulator: Accumulator, board: chess.Board, depth: int, alpha = -math.inf, beta = math.inf):
        if depth == 0 :
            stm = int(board.turn)
            with torch.no_grad(): # Improves performance slightly 
                evaluation = (stm + (stm - 1)) * net.forwardNNUE(accumulator.acc[stm], accumulator.acc[1 - stm])
                # Input white accumulator first if white to move, else input black accumulator first
            return evaluation, None
            
        maxeval = -math.inf
        bestmove = None
        # TODO move ordering, iterative deeping, tranposition table 
        for move in board.legal_moves:
            from_sq = move.from_square
            is_king_move = board.piece_at(from_sq).symbol().lower() == 'k'
            new_accumulator = accumulator.new_instance() 
            # Since depth will not be very large - depth<20 
            # a new instance of the accumulator can be created for every recursive call without taking up excessive memory
            # the new instance is needed so multiple moves can be played from one position
            if is_king_move:
                board.push(move)
                new_accumulator.refresh(board) 

            else:
                new_accumulator.update(board, move)
                board.push(move)

            eval, _ = self.search_helper(net, new_accumulator, board, depth - 1, -beta, -alpha)
            eval = -eval
            board.pop()
            
            if eval > maxeval:
                maxeval = eval
                bestmove = move
            
            alpha = max(alpha, maxeval)
            if alpha >= beta:
                break
    
        return maxeval, bestmove

    
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
    This would be the case if alpha was the best previously searched eval for white and beta is the best for black.

    However, alpha and beta could represent the best previously searched eval for the current colour and the best for the opposing colour
    then for each recursive call and increase in depth - next alpha <- -beta, next beta <- -alpha.
    as the best move for one colour is the worst move for the opposing colour.
    Using this, alpha is then a lower bound on the value since it is minus the worse move for the opposing color,
    and beta is the upper bound, since it is minus the best previously searched eval of the opposing side, and therefore 
    any position resulting in a greater evalution than beta is worse than a previously searched move for the opposing side
    and therefore the opposing side will never play the move resulting in that position.
    This means:
        when alpha > beta:
            prune position/stop searching position. 


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

def search(eval_fn, board: chess.Board, depth):
    """
    Basic Negamax search
    """
    if depth == 0 :
        return eval_fn(board), None

    maxeval = -math.inf
    bestmove = None
    for move in board.legal_moves:
        board.push(move)
        eval, _ = search(eval_fn, board, depth - 1)
        eval = -eval
        board.pop()
        if eval > maxeval:
            maxeval = eval
            bestmove = move

    return maxeval, bestmove
    
if __name__ == '__main__': #for performance profiling
    import torch 
    board = chess.Board()
    halfkp_chkpt = './nn/lightning_logs/version_halfkp/checkpoints/epoch=74-step=237300.ckpt'
    halfkp_model = nnEval_model_halfKP.load_from_checkpoint(halfkp_chkpt, net=halfKP_Net(40960, 256, 256, 256))
    net = halfkp_model.net
    searcher = SearcherNNUE(net)
    evaluation, move = searcher.search(board, 5)


