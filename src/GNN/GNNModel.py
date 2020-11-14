from collections import namedtuple

import dgl
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Constants import LABEL_NODE_NAME

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise RuntimeError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda)."
        )


from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Accuracy, Loss

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
        _, logits = self.net(graph)
        # out = logits.softmax(1)
        return logits

    def get_name(self):
        return self.__class__.__name__

    def print(self, *print_args):
        print('[%s]' % self.get_name(), end=' ')
        print(*print_args)

    @staticmethod
    def batcher(device):
        def batcher_dev(batch):
            batch_trees = dgl.batch(batch).to(device)
            return GCNBatch(graph=batch_trees,
                            labels=batch_trees.ndata[LABEL_NODE_NAME])

        return batcher_dev

    def train(self, dataset, val_dataset, optimizer=None, criterion=None, epochs=10, batch_size=1, print_step=10):
        self.net.train()

        # Collects per-epoch loss and acc.
        history = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': []}

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  collate_fn=self.batcher(self.device))

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                collate_fn=self.batcher(self.device))

        if optimizer is None:
            # create the optimizer
            optimizer = torch.optim.Adagrad(self.net.parameters(),
                                            lr=0.09,
                                            weight_decay=1e-4)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0], device=self.device))

        # training loop
        for epoch in range(epochs):
            for step, batch in enumerate(train_loader):
                g = batch.graph

                optimizer.zero_grad()
                outputs = self(g)
                loss = criterion(outputs, batch.labels)

                if torch.isnan(loss).any():
                    print(f'Loss is NAN at step: {step}')
                    continue
                loss.backward()
                optimizer.step()

                pred = torch.argmax(outputs, 1)
                acc = float(torch.sum(torch.eq(batch.labels, pred))) / len(batch.labels)
                if step % print_step == 0:
                    print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(
                        epoch, step, loss.item(), acc))

    def predict(self, dataset):
        self.net.eval()
        result = torch.empty(0, 2)

        with torch.no_grad():
            for g in dataset:
                g = g.to(self.device)
                logits = self(g)
                result = torch.cat((result, logits.cpu()), 0)
        return result

    def run(self, dataset, val_dataset, optimizer=None, criterion=None, epochs=10, batch_size=1, log_interval=10, log_dir=None):
        writer = SummaryWriter(log_dir=log_dir)

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  collate_fn=self.batcher(self.device))

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                collate_fn=self.batcher(self.device))

        if optimizer is None:
            # create the optimizer
            optimizer = torch.optim.Adagrad(self.net.parameters(),
                                            lr=0.09,
                                            weight_decay=1e-4)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0], device=self.device))

        def update_model(engine, batch):
            g = batch.graph

            optimizer.zero_grad()
            outputs = self(g)
            loss = criterion(outputs, batch.labels)

            if torch.isnan(loss).any():
                print(f'Loss is NAN ')
                return 0
            loss.backward()
            optimizer.step()
            return loss.item()

        trainer = Engine(update_model)

        val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
        evaluator = create_supervised_evaluator(self.net, metrics=val_metrics, device=self.device)

        @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(engine):
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                "".format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
            )
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics["accuracy"]
            avg_nll = metrics["nll"]
            print(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_nll
                )
            )
            writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics["accuracy"]
            avg_nll = metrics["nll"]
            print(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_nll
                )
            )
            writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

        # kick everything off
        trainer.run(train_loader, max_epochs=epochs)

        writer.close()
