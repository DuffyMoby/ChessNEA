import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from argparse import ArgumentParser
import numpy as np
import pickle as pkl
import tensorboard

chess_path = r'./Datasets/bitboardeval.pkl'
SCALING_FACTOR = 410
class BIGDATA(Dataset):
    def __init__(self, f) -> None:
        self.positions = pkl.load(f)
        self.bitboards = self.positions['bitboard']
        self.centipawns = self.positions['centipawn']
    
    def __len__(self):
        return self.centipawns.shape[0] 

    def __getitem__(self, idx):
        bitboard = self.bitboards[idx, :]
        eval = self.centipawns[idx]
        bitboard, eval = torch.from_numpy(bitboard), torch.as_tensor(eval)
        return bitboard, eval


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


class Evaluator(pl.LightningModule):
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

if __name__ == "__main__":
    with open(chess_path, 'rb') as f:
        chess_dataset = BIGDATA(f)
    train_dataloader = DataLoader(chess_dataset, batch_size=512, num_workers=0)
    # train_length = len(chess_dataset) // 1.25
    # test_length = len(chess_dataset) - train_length
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = Evaluator(SmallNet(768))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader)


