import gc
from collections import namedtuple

import dgl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Constants import LABEL_NODE_NAME
from GNN.GeneralModel import GeneralModel
from GNN.Net import Net

GCNBatch = namedtuple('GCNBatch', ['graph', 'labels'])


class GNNModel:
    def __init__(self, net, device=None):
        # self.net = Net(hidden_sizes=[8, 16, 10, 8, 4], dropout_p=dropout)
        self.net = net
        print(self.net)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Model device: {device}')
        self.device = device
        self.net.to(self.device)

    def __call__(self, graph):
        graph = graph.to(self.device)
        return self.net(graph)

    def get_name(self):
        return self.__class__.__name__

    def print(self, *print_args):
        print('[%s]' % self.get_name(), end=' ')
        print(*print_args)

    @staticmethod
    def batcher(device):
        def batcher_dev(batch):
            batch_trees = dgl.batch(batch)
            return GCNBatch(graph=batch_trees,
                            labels=batch_trees.ndata[LABEL_NODE_NAME].to(device))

        return batcher_dev

    def train(self, dataset, lr=0.09, loss_weights=None, weight_decay=1e-4, epochs=10, batch_size=1):
        self.net.train()
        if loss_weights is None:
            loss_weights = [1.0, 1.0]
        if type(loss_weights) == list:
            loss_weights = torch.FloatTensor(loss_weights).to(self.device)
        assert loss_weights.shape[0] == 2

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  collate_fn=self.batcher(self.device))
        # create the optimizer
        optimizer = torch.optim.Adagrad(self.net.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)

        # training loop
        for epoch in range(epochs):
            for step, batch in enumerate(train_loader):
                g = batch.graph.to(self.device)

                logits = self.net(g)

                logp = F.log_softmax(logits, 1)
                #         print(logp, batch.labels)
                # we only compute loss for labeled nodes
                loss = F.nll_loss(logp, batch.labels, weight=loss_weights)

                optimizer.zero_grad()
                if torch.isnan(loss).any():
                    print(f'Loss is NAN at step: {step}')
                    continue
                loss.backward()
                optimizer.step()

                pred = torch.argmax(logits, 1)
                acc = float(torch.sum(torch.eq(batch.labels, pred))) / len(batch.labels)
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
                    epoch, step, loss.item(), acc))

    def predict(self, dataset):
        self.net.eval()
        result = torch.empty(0, 2)

        with torch.no_grad():
            for g in dataset:
                g = g.to(self.device)
                logits = self.net(g)
                result = torch.cat((result, logits.cpu()), 0)
        return result
