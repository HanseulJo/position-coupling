import torch

from src.data.common import ArithmeticDataset


class CopyDataset(ArithmeticDataset):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            reverse_output=False,
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        for i in range(n_data):
            # uniform sampling of n_digit of first operand
            n_digits = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1, )).item()
            # uniform sampling of numbers a and b
            num_arr = torch.randint(0, 10, size=(n_digits,)).tolist()
            num = ''.join(map(str, num_arr))
            
            self.inputs.append(f"{num}")
            self.labels.append(f"{num}")
    

#####################################################

class CopyDatasetWithCoupledPositions(CopyDataset):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            reverse_output=False,
            randomize=True,
            max_position=30,
            vanilla=False,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        self.vanilla = vanilla # not a coupled position; vanilla randomized APE
        super().__init__(n_data, min_n_digits, max_n_digits, reverse_output, **kwargs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        start_ = 1 # (self.max_position - (len(labels)+1)) // 2 + 1
        if self.vanilla:
            pass
            start = 1 if not self.randomize else torch.randint(1, self.max_position-len(inputs)-len(labels)+1, size=(1,)).item()
            positions = list(range(start, start + len(inputs) + len(labels) + 1))
            input_positions, label_positions = positions[:len(inputs)], positions[len(inputs):]
        else:
            start = start_ if not self.randomize else torch.randint(1, self.max_position-len(inputs)+1, size=(1,)).item()
            if self.reverse_output:
                input_positions = list(range(start, start + len(inputs) + 1))
                label_positions = list(range(start, start + len(inputs)))
            else:
                input_positions = list(range(start + 1, start + len(inputs) + 1))
                label_positions = list(range(start, start + len(inputs) + 1))
        if self.reverse_output:
            labels = labels[::-1]
            label_positions = label_positions[::-1]
        # Put white spaces
        inputs = " ".join(inputs)
        labels = " ".join(labels)
        return inputs, labels, input_positions, label_positions