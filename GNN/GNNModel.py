from collections import namedtuple

import dgl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Constants import LABEL_NODE_NAME
from GNN.Net import Net

GCNBatch = namedtuple('GCNBatch', ['graph', 'labels'])


class GNNModel:
    def __init__(self, device=torch.device('cuda'), dropout=0.5, lr=0.05, weight_decay=1e-4, epochs=10, batch_size=1):
        self.net = Net(hidden_sizes=[8, 16, 10, 8, 4])
        print(self.net)

        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

    @staticmethod
    def batcher(device):
        def batcher_dev(batch):
            batch_trees = dgl.batch(batch)
            return GCNBatch(graph=batch_trees,
                            labels=batch_trees.ndata[LABEL_NODE_NAME].to(device))

        return batcher_dev

    def train(self, dataset):
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  collate_fn=self.batcher(self.device))
        # create the optimizer
        optimizer = torch.optim.Adagrad(self.net.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)

        # training loop
        for epoch in range(self.epochs):
            for step, batch in enumerate(train_loader):
                g = batch.graph

                logits = self.net(g)

                logp = F.log_softmax(logits, 1).to(self.device)
                #         print(logp, batch.labels)
                # we only compute loss for labeled nodes
                loss = F.nll_loss(logp, batch.labels)
                if torch.isnan(loss).any():
                    print(f'Loss is NAN at step: {step}')
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.argmax(logits, 1).to(self.device)
                acc = float(torch.sum(torch.eq(batch.labels, pred))) / len(batch.labels)
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
                    epoch, step, loss.item(), acc))

    def predict(self, dataset):
        pass
