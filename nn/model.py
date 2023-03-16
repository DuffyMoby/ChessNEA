import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from argparse import ArgumentParser
import pickle as pkl
import numpy as np
import tensorboard

FEATURE_SIZE_HALFKP = 40960
SCALING_FACTOR = 410

class BIGDATA(Dataset):
    def __init__(self, positions) -> None:
        self.bitboards = positions['bitboard']
        self.centipawns = positions['centipawn']
    
    def __len__(self):
        return self.centipawns.shape[0] 

    def __getitem__(self, idx):
        bitboard = self.bitboards[idx, :]
        eval = self.centipawns[idx]
        bitboard, eval = torch.from_numpy(bitboard), torch.as_tensor(eval)
        return bitboard, eval


class sparse_set1_v2(Dataset):
    def __init__(self, positions) -> None:
        self.indices = positions['indices']
        self.stms = positions['stm']
        self.centipawns = positions['centipawn']
    
    def __len__(self):
        return len(self.centipawns)

    def __getitem__(self, idx):
        indices = self.indices[idx]
        indices = np.expand_dims(indices, 0)
        values = torch.ones(indices.shape[1], dtype=torch.float32)

        eval = self.centipawns[idx]
        eval = torch.as_tensor(eval, dtype=torch.float32)

        stm = self.stms[idx]
        stm = torch.as_tensor(stm, dtype=torch.float32)

        bitboard = torch.sparse_coo_tensor(indices, values, [768])
        bitboard = bitboard.to_dense()
        
        return bitboard, eval, stm


class sparse_halfKP(Dataset):
    def __init__(self, positions) -> None:
        super().__init__()
        self.white_feature_indices = positions['white_feature_indices']
        self.black_feature_indices = positions['black_feature_indices']
        self.num_active_features = positions['num_active_features'].astype(int)
        self.stms = positions['stms'].astype(int)
        self.centipawns = positions['centipawns']
    
    def __len__(self):
        return self.centipawns.shape[0]

    def __getitem__(self, idx):
        num_active_features = self.num_active_features[idx]
        num_active_features = torch.as_tensor(num_active_features)

        white_feature_indices = self.white_feature_indices[idx]
        white_feature_indices = torch.from_numpy(white_feature_indices[:num_active_features])

        black_feature_indices = self.black_feature_indices[idx]
        black_feature_indices = torch.from_numpy(black_feature_indices[:num_active_features])

        stm = self.stms[idx]
        stm = torch.as_tensor([stm])

        eval = self.centipawns[idx]
        eval = torch.as_tensor([eval], dtype=torch.float32)

        return white_feature_indices, black_feature_indices, num_active_features, stm, eval

def collate_fn_halfkp(batch):
    white_feature_indices, black_feature_indices, num_active_features, stm, eval = zip(*batch) 
    # Dataloader returns a list of tuples from sampling dataset -> zip used to accumulate each item
    white_feature_indices = torch.cat(white_feature_indices)
    black_feature_indices = torch.cat(black_feature_indices)
    num_active_features = torch.tensor(num_active_features)
    stm = torch.stack(stm, dim=0)
    eval = torch.stack(eval, dim=0)
    # turn list of tensors into single tensor 
    batch_size = eval.shape[0]
    batch_range = torch.arange(batch_size)
    position_indices = torch.repeat_interleave(batch_range, num_active_features)
    values = torch.ones(torch.sum(num_active_features))
    white_feature_indices = torch.stack((position_indices, white_feature_indices), dim=0)
    black_feature_indices = torch.stack((position_indices, black_feature_indices), dim=0)
    # stack turns tensor into shape (2, sum(num_active_features)) so it can be input into sparse tensor
    # since sparse coordinate tensor takes indices in this form
    # indices -> [[dimension1],
    #             [dimension2]]
    # resulting in a tensor of shape (dimension1, dimension2)
    white_feature_tensor = torch.sparse_coo_tensor(
        indices=white_feature_indices,
        values=values,
        size=(batch_size, FEATURE_SIZE_HALFKP)
    )
    black_feature_tensor = torch.sparse_coo_tensor(
        indices=black_feature_indices,
        values=values,
        size=(batch_size, FEATURE_SIZE_HALFKP)
    )

    return white_feature_tensor, black_feature_tensor, stm, eval

                                                                                            
class BigNet(nn.Module):
    def __init__(self, in_size) -> None:
        super().__init__()

        self.layerStack = nn.Sequential(
            nn.Linear(in_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)
        )
    def forward(self, x):
        return self.layerStack(x)


class SmallNet(nn.Module):
    def __init__(self, in_size) -> None:
        super().__init__()

        self.layerStack = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.layerStack(x)
    

class halfKP_Net(nn.Module):
    def __init__(self, in_size, L_0_size, L_1_size, L_2_size) -> None:
        super().__init__()
        self.L_0_white = nn.Sequential(
            nn.Linear(in_size, L_0_size),
            nn.ReLU()
            )
        self.L_0_black = nn.Sequential(
            nn.Linear(in_size, L_0_size),
            nn.ReLU()
            )
        self.layerStack = nn.Sequential(
            nn.Linear(2*L_0_size, L_1_size),
            nn.ReLU(),
            nn.Linear(L_1_size, L_2_size),
            nn.ReLU(),
            nn.Linear(L_2_size, 1)
        )
    def forward(self, white_features_in, black_features_in, stm):
        w_0 = self.L_0_white(white_features_in)
        b_0 = self.L_0_black(black_features_in)
        
        acc = (stm * torch.cat([w_0, b_0], dim=1)) + ((1 - stm) * torch.cat([b_0, w_0], dim=1))
        # concatenate w_0 and b_0 such that if white to move - Layer_1 gets input [w_0, b_0], black to move - Layer_1 gets input [b_0, w_0]
        out = self.layerStack(acc)

        return out
            

class nnEval_model_set1_v1(pl.LightningModule):
    def __init__(self, net, learning_rate = 0.001) -> None:
        super().__init__()
        self.net = net
        self.lr = learning_rate
        self.lossfn = torch.nn.MSELoss()

    def training_step(self, batch):
        x, y = batch
        y = torch.reshape(y, (-1, 1)) # reshapes to tensor of size (batchsize, 1)
        wdl_eval_pred = torch.sigmoid(self.net(x) / SCALING_FACTOR) #Convert to WDL space. 
        # Sigmoid bounds values between 0-1.
        # Scaling Factor scales data such that most values lie in the range [-4,4]
        # This means large evals are much closer together
        # And there is a larger difference between more average evals found in real play. 
        wdl_eval_target = torch.sigmoid(y / SCALING_FACTOR)
        loss = self.lossfn(wdl_eval_pred, wdl_eval_target)
        self.log("train_loss", loss) # log to tensorboard
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

class nnEval_model_set1_v2(pl.LightningModule):
    def __init__(self, net, learning_rate = 0.001) -> None:
        super().__init__()
        self.net = net
        self.lr = learning_rate
        self.lossfn = torch.nn.MSELoss()

    def training_step(self, batch):
        x, y, stm = batch
        stm = torch.reshape(stm, (-1, 1)) 
        y = torch.reshape(y, (-1, 1)) 
        y_hat = self.net(x) * (stm + (stm - 1)) # negates eval if black to move

        wdl_eval_pred = torch.sigmoid(y_hat / SCALING_FACTOR) 
        wdl_eval_target = torch.sigmoid(y / SCALING_FACTOR)

        loss = self.lossfn(wdl_eval_pred, wdl_eval_target)
        self.log("train_loss", loss) # log to tensorboard
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class nnEval_model_halfKP(pl.LightningModule):
    def __init__(self, net, learning_rate = 0.001) -> None:
        super().__init__()
        self.net = net
        self.lr = learning_rate
        self.lossfn = torch.nn.MSELoss()

    def training_step(self, batch):
        white_feature_tensor, black_feature_tensor, stm, eval = batch
        eval_pred = self.net(white_feature_tensor, black_feature_tensor, stm)
        
        wdl_eval_pred = torch.sigmoid(eval_pred / SCALING_FACTOR)
        wdl_eval_target = torch.sigmoid(eval / SCALING_FACTOR)

        loss = self.lossfn(wdl_eval_pred, wdl_eval_target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

if __name__ == "__main__":
    chess_path = './Datasets/halfKP_evals_sparse.npz'
    f = open(chess_path, 'rb')
    chess_dataset_dict = np.load(f)
    # with open(chess_path, 'rb') as f:
    #     chess_dataset_dict = pkl.load(f)

    chess_dataset = sparse_halfKP(chess_dataset_dict)
    train_dataloader = DataLoader(chess_dataset, batch_size=4096, num_workers=0, collate_fn=collate_fn_halfkp)
    # train_length = len(chess_dataset) // 1.25
    # test_length = len(chess_dataset) - train_length
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = nnEval_model_halfKP(halfKP_Net(FEATURE_SIZE_HALFKP, 256, 256, 256))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader)
    f.close()


