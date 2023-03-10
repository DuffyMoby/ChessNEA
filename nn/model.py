import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
import numpy as np
import pickle as pkl
import tensorboard

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

class Sparse_set(Dataset):
    def __init__(self, positions) -> None:
        self.indices = positions['indices']
        self.stms = positions['stm']
        self.centipawns = positions['centipawn']
    
    def __len__(self):
        return len(self.centipawns)

    def __getitem__(self, idx):
        indices = self.indices[idx]
        values = torch.ones(len(indices), dtype=torch.float32)

        eval = self.centipawns[idx]
        eval = torch.as_tensor(eval, dtype=torch.float32)

        stm = torch.as_tensor(self.stms[idx], dtype=int)

        bitboard = torch.sparse_coo_tensor([indices], values, [768])
        
        return bitboard, eval, stm


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


class nnEval_model_set1_v1(pl.LightningModule):
    def __init__(self, net, learning_rate = 0.001) -> None:
        super().__init__()
        self.net = net
        self.lr = learning_rate
        self.lossfn = torch.nn.MSELoss()

    def training_step(self, batch):
        x, y = batch
        wdl_eval_pred = torch.sigmoid(self.net(x) / SCALING_FACTOR) #Convert to WDL space. 
        # Sigmoid bounds values between 0-1.
        # Scaling Factor scales data such that most values lie in the range [-4,4]
        # This means large evals are much closer together
        # And there is a larger difference between more average evals found in real play. 
        wdl_eval_target = torch.sigmoid(y / SCALING_FACTOR)
        wdl_eval_target = torch.reshape(wdl_eval_target, (-1, 1))
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
        stm = torch.reshape(stm, (-1, 1)) # converts to 2d tensor of shape (batchsize, 1)
        y = torch.reshape(y, (-1, 1)) 
        y_hat = self.net(x) * (stm + (1 - stm)) # negates eval if black to move

        wdl_eval_pred = torch.sigmoid(y_hat / SCALING_FACTOR) #Convert to WDL space. 
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

if __name__ == "__main__":
    chess_path = os.path.abspath('Datasets/bitboardeval_sparse.pkl')
    with open(chess_path, 'rb') as f:
        chess_dataset_dict = pkl.load(f)

    chess_dataset = Sparse_set(chess_dataset_dict)
    train_dataloader = DataLoader(chess_dataset, batch_size=512, num_workers=0)
    # train_length = len(chess_dataset) // 1.25
    # test_length = len(chess_dataset) - train_length
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = nnEval_model_set1_v2(SmallNet(768))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader)


