import chess
import math
from evalNNUE import Accumulator
from nn.model import halfKP_Net, nnEval_model_halfKP
import torch
import numpy as np

class SearcherNNUE:
    def __init__(self, net: halfKP_Net):
        # Get weights from network to parse into accumulator.
        # Converting weights from tensors to numpy since the operations done in the Accumulator
        # are significantly faster on ndarrays since they do not have the gradient computation complexities
        self.w_weights = net.L_0_white[0].weight.detach().numpy()
        self.w_biases = net.L_0_white[0].bias.detach().numpy()
        self.b_weights = net.L_0_black[0].weight.detach().numpy()
        self.b_biases = net.L_0_black[0].bias.detach().numpy()
        self.net = net
        self.accumulator = Accumulator(self.w_weights, self.w_biases, 
                                       self.b_weights, self.b_biases, 
                                       np.zeros((2, self.w_weights.shape[0]))) # initialize accumulator as zeros


    def search(self, board, depth, qs_depth):
        self.accumulator.refresh(board) 
        # search does not refresh the accumulator
        # therefore the accumulator must be refreshed before every search
        # so it reflects the current board state
        evaluation, move = self.search_helper(self.net, self.accumulator, board, depth, qs_depth)
        return evaluation, move


    def search_helper(self, net: halfKP_Net, accumulator: Accumulator, board: chess.Board, 
                      depth: int, qs_depth: int,  alpha = -math.inf, beta = math.inf):
        if depth == 0:
            # check quietness
            is_quiet, _ = self.is_quiet_gen_loud_moves(board)

            if is_quiet:
                # Staticly evaluate position
                stm = int(board.turn)
                # Accumulators are ndaraays, so they must be converted to tensors
                # before being input into the network
                acc1 = torch.from_numpy(accumulator.acc[stm]).float()
                acc2 = torch.from_numpy(accumulator.acc[1 - stm]).float()
                with torch.no_grad(): # Improves performance slightly 
                    evaluation = (stm + (stm - 1)) * net.forwardNNUE(acc1, acc2)
                    # Input white accumulator first if white to move, else input black accumulator first
                return evaluation, None # Best move is only relevant for the initial call
        
            else:
                return self.quiesence_search(net, accumulator, board, qs_depth, alpha, beta), None
            
        elif board.is_checkmate():
            return -50000, None
        
        bestmove = None
        for move in board.legal_moves:
            from_sq = move.from_square
            is_king_move = board.piece_at(from_sq).symbol().lower() == 'k'
            # A new accumulator instance is needed so multiple moves can be played from one position
            new_accumulator = accumulator.new_instance() 
            # Since depth will not be very large -> depth < 20, 
            # a new instance of the accumulator can be created for every recursive call 

            if is_king_move:
                board.push(move) 
                new_accumulator.refresh(board) 

            else:
                new_accumulator.update(board, move)
                board.push(move)

            score, _ = self.search_helper(net, new_accumulator, board, depth - 1, qs_depth, -beta, -alpha)
            score = -score 
            board.pop() #unmake move
            
            if score > alpha:
                alpha = score
                bestmove = move
            
            if alpha >= beta: 
                return beta, None 
    
        return alpha, bestmove
    
    def quiesence_search(self, net: halfKP_Net, accumulator: Accumulator, board: chess.Board, depth: int, alpha, beta):
        is_quiet, loud_moves = self.is_quiet_gen_loud_moves(board)
        if depth == 0 or is_quiet:
            # Staticly evaluate position
            stm = int(board.turn)

            acc1 = torch.from_numpy(accumulator.acc[stm]).float() 
            acc2 = torch.from_numpy(accumulator.acc[1 - stm]).float()

            with torch.no_grad(): 
                evaluation = (stm + (stm - 1)) * net.forwardNNUE(acc1, acc2)

            return evaluation
        
        elif board.is_checkmate():
            return -50000       
        
        for move in loud_moves: # Search only the loud moves 
            from_sq = move.from_square
            is_king_move = board.piece_at(from_sq).symbol().lower() == 'k'
            new_accumulator = accumulator.new_instance() 

            if is_king_move:
                board.push(move)
                new_accumulator.refresh(board) 

            else:
                new_accumulator.update(board, move)
                board.push(move)

            score = -self.quiesence_search(net, new_accumulator, board, depth - 1, -beta, -alpha)
            board.pop()

            alpha = max(alpha, score)

            if alpha >= beta:
                return beta
            
            
        return alpha
            

    def is_quiet_gen_loud_moves(self, board: chess.Board):
        is_check = board.is_check()
        if is_check:
            return False, board.legal_moves
        
        # checks if generate_legal_captures() is empty
        try:
            next(board.generate_legal_captures()) 
        except StopIteration:
            return True, []
        
        return False, board.generate_legal_captures()



 
def searchAB(eval_fn, board: chess.Board, depth, alpha = -math.inf, beta = +math.inf):
    """
    Negamax search with alpha-beta
    """
    if depth == 0 :
        return eval_fn(board), None
        
    bestmove = None
    for move in board.legal_moves:
        board.push(move)
        score, _ = searchAB(eval_fn, board, depth - 1, -beta, -alpha)
        score = -score
        board.pop()

        if score > alpha:
            alpha = score
            bestmove = move
        
        if alpha >= beta:
            # If best score is greater than opponents best previously searched eval
            # then stop searching node and return said previously searched eval. 
            # This guarantees this node wont be the best node of its parent node
            # since we are using strictly greater than. 
            return beta, None
        
    return alpha, bestmove

def search(eval_fn, board: chess.Board, depth):
    """
    Basic Negamax search
    """
    if depth == 0 :
        return eval_fn(board), None

    max_score = -math.inf
    bestmove = None
    for move in board.legal_moves:
        board.push(move)
        score, _ = search(eval_fn, board, depth - 1)
        score = -score
        board.pop()
        if score > max_score:
            max_score = score
            bestmove = move

    return max_score, bestmove
    
if __name__ == '__main__': #for performance profiling
    import torch 
    board = chess.Board()
    halfkp_chkpt = 'nn/lightning_logs/version_2/checkpoints/epoch=67-step=215152.ckpt'
    halfkp_model = nnEval_model_halfKP.load_from_checkpoint(halfkp_chkpt, net=halfKP_Net(40960, 256, 64, 64))
    net = halfkp_model.net
    searcher = SearcherNNUE(net)
    evaluation, move = searcher.search(board, 5, 3)


