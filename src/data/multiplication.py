import torch
from tqdm import tqdm, trange
from typing import Iterable
from src.tokenization import SpecialToken
from src.data.common import ArithmeticDataset


################ Helper Functions ################

def generate_scratchpad_multiplication(numbers:list, reversed_num=True, pad_len=(0,0)):
    A, B = numbers
    # Set pad_len_1 and pad_len_2
    if isinstance(pad_len, Iterable):
        if len(pad_len) == 2:
            pad_len_1, pad_len_2 = pad_len
        else: 
            # set both pad_len_1 and pad_len_2 as the first item of pad_len 
            pad_len = list(pad_len)[0]
            pad_len_1, pad_len_2 = pad_len, pad_len
    else:
        pad_len_1, pad_len_2 = pad_len, pad_len
    pad_len_1 = max(len(str(A)), pad_len_1)  # in order to A*0 = 00...0
    pad_len_2 = max(len(str(A)), pad_len_2)  # in order to 0+0 = 00...0
    
    # Nx1 Decomposition: Turn Multiplication into "(shifted) Addition with Multiple Operands"
    B_list = list(map(int, str(B)))[::-1]  # 105 -> [5, 0, 1]
    nby1_decomp = [A*b for b in B_list]
    nby1_decomp_str = list(map(str, nby1_decomp))
    nby1_decomp_str = ['P'*max(0, pad_len_1-len(comp)) + comp for comp in nby1_decomp_str]
    if reversed_num:
        nby1_decomp_str = [x[::-1] for x in nby1_decomp_str]
    scratchpad = '+'.join(nby1_decomp_str) + '='

    # Resolve the (shifted) Addition with Multiple Operands
    cum_sums = [0]
    mul_10 = 1
    for num in nby1_decomp:
        cum_sums.append(cum_sums[-1] + num * mul_10)
        mul_10 *= 10
    cum_sums_str = ['P'*max(0, pad_len_2-len(csum)) + csum for csum in map(str, cum_sums[1:])]
    if reversed_num:
        cum_sums_str = [x[::-1] for x in cum_sums_str]
    scratchpad += '>'.join(cum_sums_str)
    return scratchpad

################ END of Helper Functions ################

#########################################################


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
            min_n_digits_1=1,
            max_n_digits_1=3,
            min_n_digits_2=1,
            max_n_digits_2=3,
            reverse_input=False,
            reverse_output=True,
            padding=True,
            pad_token='0',
            **kwargs
        ):
        super().__init__()
        self.reverse_input = reverse_input
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        self.pad_token = pad_token
        for i in trange(n_data):
            numbers = []
            # uniform sampling of n_digits of two numbers
            n_digits_arr = [
                torch.randint(low=min_n_digits_1, high=max_n_digits_1+1, size=(1, )).item(),
                torch.randint(low=min_n_digits_2, high=max_n_digits_2+1, size=(1, )).item()
            ]
            for n_digits in n_digits_arr:
                # uniform sampling of a number
                if n_digits == 1:
                    num = torch.randint(0, 10, size=(1,)).item()
                else:
                    num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                    num = int(''.join(map(str, num_arr)))
                numbers.append(int(num))
            a, b = numbers
            result = a * b
            max_len = len(str(a)) + len(str(b))
            _inputs = [str(a), str(b)]
            if reverse_input: _inputs = [x[::-1] for x in _inputs]
            if padding:
                _labels = f"{'P'*(max_len-len(str(result)))}{result}"
            else:
                _labels = str(result)
            if reverse_output: _labels = _labels[::-1]
            _inputs = "*".join(_inputs)
            self.inputs.append(_inputs)
            self.labels.append(_labels)


#########################################################
    

class MultiplicationDatasetWithCoupledPositions(MultiplicationDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits_1=1,
            max_n_digits_1=3,
            min_n_digits_2=1,
            max_n_digits_2=3,
            reverse_input=False,
            reverse_output=False,
            padding=False,
            pad_token='0',
            randomize=True,
            max_position=22,  # max_pos >= len(operand)+2 
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        super().__init__(n_data, min_n_digits_1, max_n_digits_1, min_n_digits_2, max_n_digits_2, 
            reverse_input, reverse_output, padding, pad_token, **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        numbers = inputs.split('*')
        start = 1 if not self.randomize else torch.randint(1, self.max_position-len(labels)+1, size=(1,)).item()
        input_positions = sum((list(range(start, start+len(a)+1))[::1 if self.reverse_input else -1] for a in numbers), start=[])
        input_positions = input_positions[1:] if self.reverse_input else input_positions[:-1]
        label_positions = list(range(start, start+len(labels)+1))[::1 if self.reverse_output else -1]
        if not self.reverse_output: label_positions = label_positions[-1:] + label_positions[:-1]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))
        labels = " ".join(labels).replace('P', str(self.pad_token))
        return inputs, labels, input_positions, label_positions
    

###############################

class MultiplicationScratchPadDataset(ArithmeticDataset):
    def __init__(self,
            n_data,
            min_n_digits_1,
            max_n_digits_1,
            min_n_digits_2,
            max_n_digits_2,
            reverse_input=True,
            reverse_output=True,
            padding=True,
            pad_token='0',
            **kwargs
        ):
        self.min_n_digits_1 = min_n_digits_1
        self.max_n_digits_1 = max_n_digits_1
        self.min_n_digits_2 = min_n_digits_2
        self.max_n_digits_2 = max_n_digits_2
        self.reverse_input = reverse_input
        self.reverse_output = reverse_output
        self.pad_token = pad_token
        self.inputs = []
        self.labels = []
        for i in trange(n_data):
            numbers = []
            # uniform sampling of n_digits
            n_digits_arr = [
                torch.randint(low=min_n_digits_1, high=max_n_digits_1+1, size=(1, )).item(),
                torch.randint(low=min_n_digits_2, high=max_n_digits_2+1, size=(1, )).item()
            ]
            for n_digits in n_digits_arr:
                # uniform sampling of a number
                if n_digits == 1:
                    num = torch.randint(0, 10, size=(1,)).item()
                else:
                    num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                    num = int(''.join(map(str, num_arr)))
                numbers.append(int(num))
            
            len_sum = sum(n_digits_arr)
            result = generate_scratchpad_multiplication(
                numbers,
                reversed_num=reverse_output,
                pad_len=(n_digits_arr[0]+1, len_sum) if padding else (0,0)
            )
            _inputs = list(map(str, numbers))  # no padding in inputs
            if reverse_input: _inputs = [x[::-1] for x in _inputs]            
            _inputs = '*'.join(_inputs)
            self.inputs.append(_inputs)
            self.labels.append(result)


class MultiplicationScratchPadDatasetWithCoupledPositions(MultiplicationScratchPadDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits_1,
            max_n_digits_1,
            min_n_digits_2,
            max_n_digits_2,
            reverse_input=True,
            reverse_output=True,
            padding=True,
            pad_token='0',
            randomize=True,
            max_position_digits=50,
            max_position_operands=50,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position_digits = max_position_digits
        self.max_position_operands = max_position_operands
        self.padding = padding
        super().__init__(
            n_data,
            min_n_digits_1,
            max_n_digits_1,
            min_n_digits_2,
            max_n_digits_2,
            reverse_input,
            reverse_output,
            padding,
            pad_token,
            **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        A, B = inputs.split('*')
        lab_numbers_1, lab_numbers_2 = labels.split('=')
        lab_numbers_1 = lab_numbers_1.split('+')
        lab_numbers_2 = lab_numbers_2.split('>')
        max_len_01 = max(map(len, [A, B] + lab_numbers_1))
        max_len_02 = max(map(len, [A, B] + lab_numbers_2))

        start = 1 if not self.randomize else torch.randint(1, self.max_position_digits-max_len_01+1, size=(1,)).item()
        o = 1 if self.reverse_output else 0
        input_positions_1 = list(range(start+o, start+len(A)+o))[::1 if self.reverse_input else -1] + [0] * (len(B)+1)
        label_positions_1 = sum((list(range(start, start+len(a)+1))[::1 if self.reverse_output else -1] for a in lab_numbers_1), start=[])
        label_positions_1 += [0] * (sum(map(len, lab_numbers_2)) + len(lab_numbers_2))
        
        start = 1 if not self.randomize else torch.randint(1, self.max_position_operands-len(B)+2, size=(1,)).item()
        input_positions_2 = [0] * (len(A)+1) + list(range(start, start+len(B)))[::1 if self.reverse_input else -1]
        label_positions_2 = sum(([i] * (len(a)+1) for a, i in zip(lab_numbers_1, range(start, start+len(lab_numbers_1)))), start=[])
        label_positions_2 += sum(([j] * (len(b)+1) for b, j in zip(lab_numbers_2, range(start, start+len(lab_numbers_2)))), start=[])

        start = 1 if not self.randomize else torch.randint(1, self.max_position_digits-max_len_02+1, size=(1,)).item()
        input_positions_3 = [0] * (len(A) + 1 + len(B))
        label_positions_3 = sum((list(range(start+i, start+i+len(a)+1))[::1 if self.reverse_output else -1] for i, a in enumerate(lab_numbers_1)), start=[]) 
        label_positions_3 += sum((list(range(start, start+len(b)+1))[::1 if self.reverse_output else -1] for b in lab_numbers_2), start=[]) 

        input_positions = [input_positions_1, input_positions_2, input_positions_3]
        label_positions = [label_positions_1, label_positions_2, label_positions_3]

        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))
        labels = " ".join(labels).replace('P', str(self.pad_token))
        
        return inputs, labels, input_positions, label_positions
