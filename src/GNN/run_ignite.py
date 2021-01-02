import time
from collections import namedtuple
from datetime import datetime
from os.path import join
from typing import Union, Any, Tuple, Optional, Dict, Callable

import dgl
import torch
from ignite.handlers import EarlyStopping, DiskSaver, Checkpoint, global_step_from_engine
from torch import nn
from torch.utils.data import DataLoader

from Constants import LABEL_NODE_NAME, MODELS_PATH

# try:
#     from tensorboardX import SummaryWriter
# except ImportError:
#     try:
#         from torch.utils.tensorboard import SummaryWriter
#     except ImportError:
#         raise RuntimeError(
#             "This module requires either tensorboardX or torch >= 1.2.0. "
#             "You may install tensorboardX with command: \n pip install tensorboardX \n"
#             "or upgrade PyTorch using your package manager of choice (pip or conda)."
#         )


from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, Metric

GCNBatch = namedtuple('GCNBatch', ['graph', 'labels'])


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss


def create_my_supervised_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    def _inference(engine: Engine, batch) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            g = batch.graph
            y = batch.labels
            y_pred = model(g)
            return output_transform(g, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch).to(device)
        return GCNBatch(graph=batch_trees,
                        labels=batch_trees.ndata[LABEL_NODE_NAME])

    return batcher_dev


def run(model, dataset, val_dataset, device=None, optimizer=None, criterion=None,
        epochs=10, batch_size=1, log_interval=10,
        model_name='unknown', log_dir=None):
    start_time = time.time()

    # writer = None
    # if log_dir is not None and log_dir != '':
    #     current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    #     writer = SummaryWriter(log_dir=log_dir+current_time)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Model device: {device}')

    model.to(device)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              collate_fn=batcher(device))

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            collate_fn=batcher(device))

    if optimizer is None:
        # create the optimizer
        optimizer = torch.optim.Adagrad(model.parameters(),
                                        lr=0.09,
                                        weight_decay=1e-4)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0], device=device))

    def update_model(engine, batch):
        g = batch.graph

        optimizer.zero_grad()
        outputs = model(g)
        loss = criterion(outputs, batch.labels)

        # if torch.isnan(loss).any():
        #     print(f'Loss is NAN at step: {engine.state.iteration}')
        #     return 0
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(update_model)

    training_history = {'accuracy': [], 'loss': []}
    whole_training_history = {'loss': []}
    validation_history = {'accuracy': [], 'loss': []}
    val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    train_evaluator = create_my_supervised_evaluator(model, metrics=val_metrics)
    val_evaluator = create_my_supervised_evaluator(model, metrics=val_metrics)

    handler = EarlyStopping(patience=50, score_function=score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    to_save = {'model': model}
    handler = Checkpoint(to_save, DiskSaver(join(MODELS_PATH, model_name), create_dir=True, require_empty=False),
                         n_saved=2, score_function=score_function, score_name="loss",
                         global_step_transform=global_step_from_engine(trainer))

    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        whole_training_history['loss'].append(engine.state.output)
        # if writer is not None:
        #     writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        print(f'Train results | '
              f'Epoch {engine.state.epoch:05d} | '
              f'Avg loss {avg_loss:.4f} | '
              f'Avg accuracy {avg_accuracy:.4f} |')
        training_history['accuracy'].append(avg_accuracy)
        training_history['loss'].append(avg_loss)
        # if writer is not None:
        #     writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        #     writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        # print(f'Val results   | '
        #       f'Epoch {engine.state.epoch:05d} | '
        #       f'Avg loss {avg_loss:.4f} | '
        #       f'Avg accuracy {avg_accuracy:.4f} |')
        validation_history['accuracy'].append(avg_accuracy)
        validation_history['loss'].append(avg_loss)
        # if writer is not None:
        #     writer.add_scalar("valdation/avg_loss", avg_loss, engine.state.epoch)
        #     writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    # if writer is not None:
    #     writer.close()

    print(f'Model trained in {(time.time() - start_time):.1f}s')
    return training_history, validation_history, whole_training_history
