import os
import pandas as pd
from math import exp
import numpy as np
from tqdm import trange
import pickle
data_path = r"./Datasets/chessData.csv"
ONETOEIGHT = set(map(str, range(1,9)))
PIECE_TABLE = {
    'P' : (0,1),
    'p' : (1,-1),
    'N' : (2,1),
    'n' : (3,-1),
    'B' : (4,1),
    'b' : (5,-1),
    'R' : (6,1),
    'r' : (7,-1),
    'Q' : (8,1),
    'q' : (9,-1),
    'K' : (10,1),
    'k' : (11,-1)
}

COLOR_TABLE = {
    'w' : 1,
    'b' : -1
}

SCALING_FACTOR = 410

def sigmoid(x):
    return 1/(1+exp(-x))

def fentobitboard(fen):
    board, color, *_ = fen.split()
    board = board.replace("/", "")
    sq = 0
    out = np.zeros(768, dtype=np.float32)
    for pc in board:
        if pc in ONETOEIGHT:
            sq += int(pc)
        else:
            idx = PIECE_TABLE[pc][0] * 64 + sq
            out[idx] = COLOR_TABLE[color] * PIECE_TABLE[pc][1] 
            # Changes sign of bit such that an opposing piece has value -1 and friendly piece has value 1 in bitboard
            # This is to establish side to move in the bitboard without adding an extra bit
            sq += 1
    return out

def transformCP(val):
    if val[0] == '#':
        return int(val[1] + '5000')
        
    elif val == '\ufeff+23': # For whatever reason, some zeros have value '\ufeff+23' 
        return 0

    else:
        return int(val)


if __name__ == '__main__':
    # bigdict = {
    #     'centipawn' : [],
    #     'bitboard' : []
    # }
    chess_data = pd.read_csv(data_path)
    # for x in trange(len(chess_data)//2):
    #     centipawn = transformCP(chess_data.iloc[x, 1])
    #     bitboard = fentobitboard(chess_data.iloc[x, 0])

    #     bigdict['centipawn'].append(centipawn)
    #     bigdict['bitboard'].append(bitboard)

    # transformed_data = pd.DataFrame(data=bigdict)
    # transformed_data.to_pickle('./Datasets/bitboardswitheval.pkl')

    num_entries = len(chess_data)
    big = {
        'centipawn' : np.ndarray((num_entries), dtype=np.float32),
        'bitboard' : np.ndarray((num_entries, 768), dtype=np.float32)
    }
    for i in trange(num_entries):
        centipawn = transformCP(chess_data.iloc[i, 1])
        bitboard = fentobitboard(chess_data.iloc[i, 0])

        big['centipawn'][i] = centipawn
        big['bitboard'][i] = bitboard

    filepath = r'./Datasets/bitboardeval.pkl'

    with open(filepath, 'wb') as f:
        pickle.dump(big, f)


    



    



