from abc import ABC, abstractmethod


class GeneralModel(ABC):
    @abstractmethod
    def train(self, dataset, lr, weight_decay, epochs, batch_size):
        """
            Train model
        :param dataset: list of data
        :param lr: learning rate
        :param weight_decay: weight decay
        :param epochs: number of epochs
        :param batch_size: batch size
        """
        pass

    @abstractmethod
    def predict(self, dataset):
        """
            Predict values from dataset
        :param dataset: list of data
        :return: predicted sentiments (list of dictionaries)
        """
        return None

    def get_name(self):
        return self.__class__.__name__

    def print(self, *print_args):
        print('[%s]' % self.get_name(), end=' ')
        print(*print_args)
