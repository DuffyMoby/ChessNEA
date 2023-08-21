import chess
from chess import InvalidMoveError
import search
from nn.model import nnEval_model_halfKP, halfKP_Net
COLOR_TABLE = {
    0: 'Black',
    1: 'White'
}


def getMove(board: chess.Board) -> chess.Move:
    move = input("Enter your move in UCI format. eg. g1f3| q to quit ")
    if move == 'q':
        return move
    # Checks if move is in valid UCI format
    try:
        move = chess.Move.from_uci(move)
    except InvalidMoveError:
        print('Invalid move.')
        move = getMove(board)

    if move not in board.legal_moves:
        print('Illegal move')
        move = getMove(board)

    return move

def evaluation_from_FEN():
    fen = input("Input FEN: ")
    depth = int(input('Input depth of engine search. (Integer): '))
    qs_depth = int(input('Input quiescence search depth. (Integer): '))

    board = chess.Board(fen=fen)
    halfkp_chkpt = './nn/lightning_logs/version_2/checkpoints/epoch=67-step=215152.ckpt'
    halfkp_model = nnEval_model_halfKP.load_from_checkpoint(halfkp_chkpt, net=halfKP_Net(40960, 256, 64, 64))
    net = halfkp_model.net
    searcher = search.SearcherNNUE(net)
    evaluation, move = searcher.search(board, depth, qs_depth)
    print(f'Evaluation: {evaluation.item()}| Best move: {move.uci()}')

# Litterally 1-8 but as strings
ONETOEIGHT = list(map(str,range(1,9)))

def printBoard(board: chess.Board, perspective):
    board, *_ = board.fen().split()
    board = board.split('/')
    for i, col in enumerate(board) if perspective == 1 else list(enumerate(board))[::-1]: # iterate through board upside down if black
        print(8 - i, end='') # print row number
        for piece in col if perspective == 1 else col[::-1]: # columns also have to be reversed if black since board is rotated
            if piece in ONETOEIGHT:
                # print number times " ."
                print(int(piece) * " ·", end='')
            else:
                # print unicode representation of piece
                print(f' {chess.UNICODE_PIECE_SYMBOLS[piece]}', end='')

        print()
    AtoH = [chr(i) for i in range(65, 73)]
    print("  " + " ".join(AtoH if perspective == 1 else AtoH[::-1])) # prints column letters


def main():
    fen = input("Enter starting board FEN. Press Enter to start normally. ")
    depth = int(input('Input depth of engine search. (Integer): '))
    qs_depth = int(input('Input quiescence search depth. (Integer): '))
    # Get player side
    player_side = input("Enter side to move. (1)white (0)black: ")
    while not (player_side == '0' or player_side == '1'):
        player_side = input("Enter side to move. (1)white (0)black: ")
    player_side = int(player_side)
    # Initilise board object
    if fen:
        board = chess.Board(fen)
    else:
        board = chess.Board()

    # Initialize model and searcher 
    halfkp_chkpt = './nn/lightning_logs/version_2/checkpoints/epoch=67-step=215152.ckpt'
    halfkp_model = nnEval_model_halfKP.load_from_checkpoint(halfkp_chkpt, net=halfKP_Net(40960, 256, 64, 64))
    net = halfkp_model.net
    searcher = search.SearcherNNUE(net)

    printBoard(board, player_side)
    while True:
        # If player to move get move from player
        if board.turn == player_side:
            move = getMove(board)

        else: # Otherwise use engine move
            evaluation, move = searcher.search(board, depth, qs_depth)
            print(f'Evaluation: {evaluation.item()}')

        if move == 'q':
            break

        board.push(move) # play move
        printBoard(board, player_side)

        if board.is_checkmate():
            print(COLOR_TABLE[1 - board.turn], 'wins')
            break

        if board.is_stalemate():
            print('Draw')
            break
    
    print(f'End board FEN: {board.fen()}')


if __name__ == '__main__':
    main()

        


