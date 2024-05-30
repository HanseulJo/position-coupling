import torch
from tqdm import tqdm, trange

from src.tokenization import SpecialToken
from src.data.common import ArithmeticDataset


class NxMMultiplicationDataset(ArithmeticDataset):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            M=1, # M is a fixed length of second operand
            reverse_output=False,
            padding=False,
            pad_token=SpecialToken.pad,
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.pad_token = pad_token
        self.inputs = []
        self.labels = []
        for i in trange(n_data):
            # uniform sampling of n_digit of first operand
            n_digits = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1, )).item()
            # uniform sampling of numbers a and b
            if n_digits == 1:
                a = torch.randint(0, 10, size=(1,)).item()
            else:
                num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                a = int(''.join(map(str, num_arr)))
            if M == 1:
                b = torch.randint(low=0, high=10, size=(1,)).item()
            else:
                num_arr2 = torch.randint(0, 10, size=(M,)).tolist()
                b = int(''.join(map(str, num_arr2)))
            result = a * b
            len_sum = n_digits + M
            self.inputs.append(f"{a}*{b:0{M}d}")
            self.labels.append(f"{'P'*(len_sum-len(str(result)))}{result}" if padding else f"{result}")


#############################################################################


class NxMMultiplicationDatasetWithCoupledPositions(NxMMultiplicationDataset):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            M=1,
            reverse_output=False,
            padding=False,
            pad_token=SpecialToken.pad,
            randomize=True,
            max_position=20,
            vanilla=False,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        self.vanilla = vanilla
        super().__init__(n_data, min_n_digits, max_n_digits, M, reverse_output, padding, pad_token, **kwargs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        a, b = inputs.split('*')
        max_len = len(a)
        start_ = 1 #(self.max_position - (len(labels)+1)) // 2 + 1
        if self.vanilla:
            start = 1 if not self.randomize else torch.randint(1, self.max_position-len(inputs)-len(labels)+1, size=(1,)).item()
            positions = list(range(start, start + len(inputs) + len(labels) + 1))
            input_positions, label_positions = positions[:len(inputs)], positions[len(inputs):]
        else:
            start = start_ if not self.randomize else torch.randint(1, self.max_position-max_len, size=(1,)).item()
            end = start + max_len + len(b)
            input_positions = list(range(end-len(a), end+1)) + list(range(end-len(b), end+1))
            label_positions = list(range(end-len(labels), end))
        if self.reverse_output:
            labels = labels[::-1]
            if not self.vanilla:
                label_positions = label_positions[::-1]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))
        labels = " ".join(labels).replace('P', str(self.pad_token))
        return inputs, labels, input_positions, label_positions
    

#########################################################


class MultiplicationDataset(ArithmeticDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token=SpecialToken.pad,
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        self.pad_token = pad_token
        for i in range(n_data):
            if len(self.inputs) >= n_data: break
            numbers = []
            # uniform sampling of n_digits of two numbers
            n_digits_arr = torch.randint(low=min_n_digits, 
                                        high=max_n_digits+1, size=(2, ))
            for i in range(2):
                n_digits = n_digits_arr[i].item()
                # uniform sampling of a number
                if n_digits == 1:
                    num = torch.randint(0, 10, size=(1,)).item()
                else:
                    num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                    num = int(''.join(map(str, num_arr)))
                numbers.append(int(num))
            a, b = numbers
            result = a * b
            max_len = max(len(str(a)),len(str(b)))
            self.inputs.append(f"{'P'*(max_len-len(str(a)))}{a}*{'P'*(max_len-len(str(b)))}{b}" 
                               if padding else f"{a}+{b}")
            self.labels.append(f"{'P'*(max_len+1-len(str(result)))}{result}" 
                               if padding else f"{result}")
            if commutative and a != b:
                self.inputs.append(f"{'P'*(max_len-len(str(b)))}{b}*{'P'*(max_len-len(str(a)))}{a}" 
                                   if padding else f"{b}+{a}")
                self.labels.append(f"{'P'*(max_len+1-len(str(result)))}{result}" 
                                   if padding else f"{result}")


#########################################################
    

class MultiplicationDatasetWithCoupledPositions(MultiplicationDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token='0',
            randomize=True,
            max_position=22,  # max_pos >= len(operand)+2 
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        super().__init__(n_data, min_n_digits, max_n_digits,
            reverse_output, commutative, padding, pad_token, **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        a, b = inputs.split('*')
        max_len = max(len(a),len(b))
        start = 1 if not self.randomize else torch.randint(1, self.max_position-max_len, size=(1,)).item()
        end = start + max_len + 1
        input_positions = list(range(end-len(a), end+1)) + list(range(end-len(b), end+1))
        label_positions = list(range(end-len(labels), end))
        if self.reverse_output:
            labels = labels[::-1]
            label_positions = label_positions[::-1]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))
        labels = " ".join(labels).replace('P', str(self.pad_token))
        return inputs, labels, input_positions, label_positions
    
