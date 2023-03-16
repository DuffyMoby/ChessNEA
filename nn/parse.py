import pandas as pd
import numpy as np
from tqdm import trange
import pickle
from copy import deepcopy

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
    board = board.split("/")[::-1]
    board = "".join(board)
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


def fen2bitboard_sparse_set1_v2(fen):
    board, color, *_ = fen.split()
    board = board.split("/")[::-1]
    board = "".join(board)
    sq = 0
    out_indices = []
    for pc in board:
        if pc in ONETOEIGHT:
            sq += int(pc)
        else:
            idx = PIECE_TABLE[pc][0] * 64 + sq
            sq += 1
            out_indices.append(idx)

    return np.array(out_indices, dtype=np.int64), COLOR_TABLE[color]


def fen2halfKP_sparse(fen, max_active_features):
    board, color, *_ = fen.split()
    board = board.split("/")[::-1]
    board = "".join(board)
    w_king_sq = 0
    b_king_sq = 0
    sq = 0
    num_active_features = 0
    piece_indices = np.zeros(max_active_features, dtype=np.int64)
    for pc in board:
        if pc == 'K':
            w_king_sq = sq
            sq += 1

        elif pc == 'k':
            b_king_sq = sq
            sq += 1

        elif pc in ONETOEIGHT:
            sq += int(pc)

        else:
            idx = PIECE_TABLE[pc][0] * 64 + sq
            sq += 1
            piece_indices[num_active_features] = idx
            num_active_features += 1

    white_indices = deepcopy(piece_indices)
    black_indices = deepcopy(piece_indices)

    white_indices[:num_active_features] = white_indices[:num_active_features] + (w_king_sq * 10) * 64
    black_indices[:num_active_features] = black_indices[:num_active_features] + (b_king_sq * 10) * 64
    return white_indices, black_indices, num_active_features, COLOR_TABLE[color]
    
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
    Stores whole bitboard input along with its eval
    Results in a rather unnecesarily large file -> 40GB!!!
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


def main_sparse_set1_v2(fen_transform, data_path, out_path):
    """
    For second version of feature set 1
    Stores indices of non zero bits (only ones)
    Results in much smaller file size since feature set is very sparse (<5% non zero values)
    """
    chess_data = pd.read_csv(data_path)
    num_entries = len(chess_data)
    d = {
        'indices' : np.ndarray(num_entries, dtype=np.object_),
        'stm' : np.ndarray(num_entries), # np arrays are marginally faster to load
        'centipawn' : np.ndarray(num_entries, dtype=np.float32)
    }
    for i in trange(num_entries):
        centipawn = transformCP(chess_data.iloc[i, 1])
        indices, stm = fen_transform(chess_data.iloc[i, 0])

        d['indices'][i] = indices
        d['stm'][i] = stm
        d['centipawn'][i] = centipawn

    with open(out_path, 'wb') as f:
        pickle.dump(d, f)

def main_sparse_HALFKP(fen_transform, data_path, out_path):
    """
    For second version of feature set 1
    Stores indices of non zero bits (only ones)
    Results in much smaller file size since feature set is very sparse (<5% non zero values)
    """
    chess_data = pd.read_csv(data_path)
    num_entries = len(chess_data)
    max_features = 30

    white_feature_indices = np.ndarray((num_entries, max_features), dtype=np.int64)
    black_feature_indices = np.ndarray((num_entries, max_features), dtype=np.int64)
    num_active_features = np.ndarray(num_entries, dtype=int)
    stms = np.ndarray(num_entries, dtype=int) # np arrays are marginally faster to load
    centipawns = np.ndarray(num_entries, dtype=np.float32)
    
    for i in trange(num_entries):
        centipawn = transformCP(chess_data.iloc[i, 1])
        white_indices, black_indices, num_active, stm = fen_transform(chess_data.iloc[i, 0], max_features)

        white_feature_indices[i] = white_indices
        black_feature_indices[i] = black_indices
        num_active_features[i] = num_active
        stms[i] = stm
        centipawns[i] = centipawn

    with open(out_path, 'wb') as f:
        np.savez(f, white_feature_indices=white_feature_indices, black_feature_indices=black_feature_indices,num_active_features=num_active_features ,stms=stms, centipawns=centipawns)



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
    out_path = r'./Datasets/halfKP_evals_sparse.npz'
    main_sparse_HALFKP(fen_transform=fen2halfKP_sparse, data_path=data_path,out_path=out_path)


    



    



