import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenization import BINARY_OPS, SpecialToken


OPERATORS = {
    '+':    {"n_input": 2, "operation": lambda x: x[0]+x[1]},
    '-':    {"n_input": 2, "operation": lambda x: x[0]-x[1]},
    '*':    {"n_input": 2, "operation": lambda x: x[0]*x[1]},
    '/':    {"n_input": 2, "operation": lambda x: x[0]/x[1]},
    '//':   {"n_input": 2, "operation": lambda x: x[0]//x[1]},
    '%':    {"n_input": 2, "operation": lambda x: x[0]%x[1]},
    'max':  {"n_input": 2, "operation": lambda x: max(x[0], x[1])},
    'min':  {"n_input": 2, "operation": lambda x: min(x[0], x[1])},
    'sin':  {"n_input": 1, "operation": np.sin},
    'cos':  {"n_input": 1, "operation": np.cos},
    'sqrt': {"n_input": 1, "operation": np.sqrt},
    'copy': {"n_input": 1, "operation": lambda x: x},
    'parity': {"n_input": 1, "operation": lambda x: sum(map(int, str(x))) % 2},
    None: {"n_input": None, "operation": None},
}


class Operation:
    def __init__(self, symbol, n_input, operation):
        self.symbol = symbol
        self.n_input = n_input
        self.operation = operation
    
    def __call__(self, args):
        if self.n_input > 2:
            assert len(args) == self.n_input
        return self.operation(args)


## Parent Class of All Synthetic Dataset 
class ArithmeticDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.labels = []
        self.reverse_input = False
        self.reverse_output = False
        self.pad_token = SpecialToken.pad  # '0'

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))  # Converts 'P' --> pad_token
        labels = " ".join(labels).replace('P', str(self.pad_token))  # Converts 'P' --> pad_token
        return inputs, labels
