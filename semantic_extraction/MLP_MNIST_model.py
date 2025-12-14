#!/usr/bin/env python
# encoding: utf-8
"""
Train and save the MNIST classifier (MLP).
Adds argparse, deterministic seed handling and clearer dataset/save paths.
"""
import argparse
import logging
import random
from pathlib import Path

import torch
import numpy as np
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST MLP classifier')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default=None, help='directory to save models')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Paths: dataset lives next to this script under dataset/, models saved to repo-level saved_model/
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_SAVE_DIR = BASE_DIR.parent / 'saved_model'
    SAVE_DIR = Path(args.save_dir) if args.save_dir else DEFAULT_SAVE_DIR
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def data_transform(x):
        x = np.array(x, dtype='float32') / 255
        x = x.reshape((-1,))
        x = torch.from_numpy(x)
        return x

    # load data (dataset directory relative to this script)
    dataset_dir = BASE_DIR / 'dataset' / 'mnist'
    trainset = mnist.MNIST(str(dataset_dir), train=True, transform=data_transform, download=True)
    testset = mnist.MNIST(str(dataset_dir), train=False, transform=data_transform, download=True)
    train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    class MLP(nn.Module):
        # classifier
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 500)
            self.fc2 = nn.Linear(500, 250)
            self.fc3 = nn.Linear(250, 125)
            self.fc4 = nn.Linear(125, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    mlp = MLP()

    criterion = nn.CrossEntropyLoss()

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for e in range(args.epochs):
        # SGD or Adam
        if e < 7:
            optimizer = torch.optim.Adam(mlp.parameters(), 1e-3)
        else:
            optimizer = torch.optim.Adam(mlp.parameters(), 1e-4)

        train_loss = 0
        train_acc = 0
        mlp.train()
        for im, label in train_data:
            im = Variable(im)
            label = Variable(label)
            # forward
            out = mlp(im)

            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))

        eval_loss = 0
        eval_acc = 0
        mlp.eval()
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)
            out = mlp(im)
            loss = criterion(out, label)

            eval_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(e, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data)))

    model_path = SAVE_DIR / 'MLP_MNIST.pkl'
    torch.save(mlp.state_dict(), str(model_path))
    logging.info(f"Saved model to {model_path}")

    # Optionally save run config (if caller requested)
    try:
        from utils.save_run import save_run_config
    except Exception:
        save_run_config = None

    # args may contain save_config and output_dir when used from CLI
    if hasattr(args, 'save_config') and getattr(args, 'save_config'):
        cfg = {
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'test_batch_size': args.test_batch_size,
            'save_dir': str(SAVE_DIR),
            'model_path': str(model_path),
        }
        out_dir = Path(args.save_dir) if args.save_dir else BASE_DIR.parent / 'results'
        if save_run_config:
            p = save_run_config(cfg, out_dir)
            logging.info(f'Saved training config to {p}')
        else:
            # fallback: write basic JSON here
            out_dir.mkdir(parents=True, exist_ok=True)
            import json
            p = out_dir / 'run_config_train.json'
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            logging.info(f'Saved training config to {p}')

    # file = './results/MLP_MNIST_model/acc.csv'
    # data = pd.DataFrame(eval_acces)
    # data.to_csv(file, index=False)
    #
    # file = './results/MLP_MNIST_model/loss.csv'
    # data = pd.DataFrame(eval_losses)
    # data.to_csv(file, index=False)


if __name__ == '__main__':
    main()






