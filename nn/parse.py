import pandas as pd
from math import exp
import numpy as np
from tqdm import trange
import pickle

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
    'b' : 0
}

def fen2bitboard(fen):
    board, color, *_ = fen.split()
    board = board.replace("/", "")
    sq = 0
    out = np.zeros(768, dtype=np.float32)
    for pc in board:
        if pc in ONETOEIGHT:
            sq += int(pc)
        else:
            idx = PIECE_TABLE[pc][0] * 64 + sq
            out[idx] = PIECE_TABLE[pc][1] * (COLOR_TABLE[color] + (COLOR_TABLE[color] - 1))
            # Changes sign of bit such that an opposing piece has value -1 and friendly piece has value 1 in bitboard
            sq += 1
    return out


def fen2bitboard_sparse(fen):
    board, color, *_ = fen.split()
    board = board.replace("/", "")
    sq = 0
    out_indices = []
    for pc in board:
        if pc in ONETOEIGHT:
            sq += int(pc)
        else:
            idx = PIECE_TABLE[pc][0] * 64 + sq
            sq += 1
            out_indices.append(idx)

    return out_indices, COLOR_TABLE[color]

    

def transformCP(val):
    if val[0] == '#':
        return int(val[1] + '5000')
        
    elif val == '\ufeff+23': # For whatever reason, some zeros have value '\ufeff+23' 
        return 0

    else:
        return int(val)


def main_set1_v1(data_path, out_path):
    """
    For first version of feature set 1
    """
    chess_data = pd.read_csv(data_path)
    num_entries = len(chess_data)
    big = {
        'centipawn' : np.ndarray((num_entries), dtype=np.float32),
        'bitboard' : np.ndarray((num_entries, 768), dtype=np.float32)
    }
    for i in trange(num_entries):
        centipawn = transformCP(chess_data.iloc[i, 1])
        bitboard = fen2bitboard(chess_data.iloc[i, 0])

        big['centipawn'][i] = centipawn
        big['bitboard'][i] = bitboard

    with open(out_path, 'wb') as f:
        pickle.dump(big, f)


def main_sparse_set1_v2(data_path, out_path):
    """
    For second version of feature set 1
    """
    chess_data = pd.read_csv(data_path)
    num_entries = len(chess_data)
    d = {
        'indices' : [],
        'stm' : np.ndarray(num_entries), # np arrays are marginally faster to load
        'centipawn' : np.ndarray(num_entries, dtype=np.float32)
    }
    for i in trange(num_entries):
        centipawn = transformCP(chess_data.iloc[i, 1])
        indices, stm = fen2bitboard_sparse(chess_data.iloc[i, 0])

        d['indices'].append(indices)
        d['stm'][i] = stm
        d['centipawn'][i] = centipawn

    with open(out_path, 'wb') as f:
        pickle.dump(d, f)



if __name__ == '__main__':
    # bigdict = {
    #     'centipawn' : [],
    #     'bitboard' : []
    # }
    # for x in trange(len(chess_data)//2):
    #     centipawn = transformCP(chess_data.iloc[x, 1])
    #     bitboard = fentobitboard(chess_data.iloc[x, 0])

    #     bigdict['centipawn'].append(centipawn)
    #     bigdict['bitboard'].append(bitboard)

    # transformed_data = pd.DataFrame(data=bigdict)
    # transformed_data.to_pickle('./Datasets/bitboardswitheval.pkl')
    data_path = r"./Datasets/chessData.csv"
    out_path = r'./Datasets/bitboardeval_sparse.pkl'
    main_sparse_set1_v2(data_path=data_path,out_path=out_path)


    



    



