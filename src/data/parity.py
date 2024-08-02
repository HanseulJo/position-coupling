import torch

from src.data.common import Operation, ArithmeticDataset


################ Helper Functions ################


def generate_scratchpad_for_parity(string):
    scratchpad = string[0]
    for char in string[1:]:
        scratchpad += str(abs(int(char)-int(scratchpad[-1])))
    return scratchpad


################ END of Helper Functions ################

#########################################################


class ParityDataset(ArithmeticDataset):
    def __init__(self, 
            operation:Operation, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            **kwargs
        ):
        super().__init__()
        self.operation = operation
        self.inputs = []
        self.labels = []
        if not (operation.n_input == 1 and operation.symbol in ['parity']):
            raise ValueError("operation.symbol != 'parity'")
        for i in range(n_data):
            # uniform sampling of n_digit of first operand
            n_digits = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1, )).item()
            # uniform sampling of binary string
            num = ''.join(map(str, torch.randint(low=0, high=2, size=(n_digits,)).tolist()))
            self.inputs.append(f"{num}")
            self.labels.append(f"{operation(num)}")


#########################################################


class ParityDatasetWithCoupledPositions(ParityDataset):
    def __init__(self, 
            operation:Operation, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            randomize=True,
            max_position=30,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        super().__init__(operation, n_data, min_n_digits, max_n_digits, **kwargs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        start = 1 if not self.randomize else torch.randint(1, self.max_position, size=(1,)).item()
        input_positions = [start] * len(inputs) + [start+1]
        label_positions = [start] * len(labels)
        # Put white spaces
        inputs = " ".join(inputs)
        labels = " ".join(labels)
        return inputs, labels, input_positions, label_positions


#########################################################


class ParityDatasetWithScratchpad(ArithmeticDataset):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            reversed_scratchpad=False,
            **kwargs
        ):
        super().__init__()
        self.reversed_scratchpad = reversed_scratchpad
        self.inputs = []
        self.labels = []
        for i in range(n_data):
            # uniform sampling of n_digit of first operand
            n_digits = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1, )).item()
            # uniform sampling of binary string
            num = ''.join(map(str, torch.randint(low=0, high=2, size=(n_digits,)).tolist()))
            if self.reversed_scratchpad:
                scratchpad = generate_scratchpad_for_parity(num[::-1])
            else:
                scratchpad = generate_scratchpad_for_parity(num)
            self.inputs.append(num)
            self.labels.append(scratchpad)


#########################################################


class ParityDatasetWithScratchpadAndCoupledPositions(ParityDatasetWithScratchpad):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            reversed_scratchpad=False,
            randomize=True,
            max_position=30,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        super().__init__(n_data, min_n_digits, max_n_digits, reversed_scratchpad=reversed_scratchpad, **kwargs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        start_ = 1 #(self.max_position - (len(labels)+1)) // 2 + 1
        start = start_ if not self.randomize else torch.randint(1, self.max_position-len(inputs)+1, size=(1,)).item()
        
        if self.reversed_scratchpad:
            # naive
            input_positions = list(range(start, start+len(inputs)))
            label_positions = list(range(start, start+len(labels)+1))[::-1]
            # construction
            # input_positions = list(range(start+1, start + len(inputs) + 1))
            # label_positions = list(range(start, start + len(labels) + 1))[::-1]
        else:
            # naive
            input_positions = list(range(start+1, start+1+len(inputs)))
            label_positions = list(range(start, start+len(labels)+1))
            # construction
            # input_positions = list(range(start, start + len(inputs)))
            # label_positions = list(range(start, start + len(labels) + 1))
        # Put white spaces
        inputs = " ".join(inputs)
        labels = " ".join(labels)
        return inputs, labels, input_positions, label_positions