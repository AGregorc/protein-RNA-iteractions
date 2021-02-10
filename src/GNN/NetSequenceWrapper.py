import typing

from torch import nn


class NetSequenceWrapper(nn.Module):
    def __init__(self, *sequence):
        super(NetSequenceWrapper, self).__init__()
        self.sequence = nn.Sequential(*sequence)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, g, *args):
        _, output = self.sequence((g, *args))
        return output
