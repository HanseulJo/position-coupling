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
        if self.reverse_input:
            inputs = labels[::-1]
        if self.reverse_output:
            labels = labels[::-1]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))  # Converts 'P' --> pad_token
        labels = " ".join(labels).replace('P', str(self.pad_token))  # Converts 'P' --> pad_token
        return inputs, labels


class ArithmeticDatasetUniformSampling(ArithmeticDataset):
    def __init__(self, 
            operation:Operation, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            reverse_output=False,
            commutative=False,
            padding=False, # dynamic zero padding
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        for i in range(n_data):
            numbers = []
            # uniform sampling of n_digits of two numbers
            n_digits_arr = torch.randint(low=min_n_digits, 
                                         high=max_n_digits+1, size=(2, ))
            for i in range(operation.n_input):
                n_digits = n_digits_arr[i].item()
                # uniform sampling of a number
                if n_digits == 1:
                    num = torch.randint(0, 10, size=(1,)).item()
                else:
                    num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                    num = int(''.join(map(str, num_arr)))
                # num = torch.randint(low=10**(n_digits-1) - (1 if n_digits==1 else 0), high=10**n_digits, size=(1,)).item()
                numbers.append(int(num))
            
            if operation.n_input == 2 and operation.symbol in ['+']:
                a, b = numbers
                _max_len = max(len(str(a)), len(str(b)))
                self.inputs.append(f"{a:0{_max_len}d}{operation.symbol}{b:0{_max_len}d}" if padding else f"{a}{operation.symbol}{b}")
                self.labels.append(f"{operation([a,b])}")
                if commutative and a != b:
                    self.inputs.append(f"{b:0{_max_len}d}{operation.symbol}{a:0{_max_len}d}" if padding else f"{b}{operation.symbol}{a}")
                    self.labels.append(f"{operation([b,a])}")
            elif operation.n_input == 2 and operation.symbol in ['*']:
                a, b = numbers
                self.inputs.append(f"{a}{operation.symbol}{b}")
                self.labels.append(f"{operation([a,b])}")
                if commutative and a != b:
                    self.inputs.append(f"{b}{operation.symbol}{a}")
                    self.labels.append(f"{operation([b,a])}")
            elif operation.n_input == 2 and operation.symbol in BINARY_OPS:
                a, b = numbers
                self.inputs.append(f"{a}{operation.symbol}{b}")
                self.labels.append(f"{operation([a,b])}")
            elif operation.n_input == 1: 
                # self.inputs.append(f"{operation.symbol}({numbers[0]})".replace(' ',''))
                self.inputs.append(f"{numbers[0]}")
                self.labels.append(f"{operation(numbers[0])}")
            else:
                self.inputs.append(f"{operation.symbol}{tuple(numbers)}".replace(' ',''))
                self.labels.append(f"{operation(numbers)}")
