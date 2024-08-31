import torch
import numpy as np
from tqdm import tqdm, trange
from src.tokenization import SpecialToken
from src.data.common import ArithmeticDataset


################ Helper Functions ################


def generate_hard_carry_sum_pair(n_digits):
    a = 0
    b = 0
    for j in range(n_digits):
        if j == n_digits-1:
            a_digit = torch.randint(low=1, high=10, size=(1,)).item()
            b_digit = torch.randint(low=10-a_digit, high=10, size=(1,)).item()
        else:
            a_digit = torch.randint(low=0, high=10, size=(1,)).item()
            b_digit = 9 - a_digit
        a = 10 * a + a_digit
        b = 10 * b + b_digit
    return a, b


def get_carries_for_addition(a, b):
    _max_len = max(len(str(a)), len(str(b)))
    a = f"{a:0{_max_len}d}"
    b = f"{b:0{_max_len}d}"

    carries = "0"
    direct_carries = "0"
    virtual_carries = "0"
    for i, (ai, bi) in enumerate(zip(a[::-1], b[::-1])):
        prev = carries[-1]
        if int(ai) + int(bi) + int(prev) >= 10:
            carries += '1'
            if int(ai) + int(bi) == 9 and int(prev) == 1:
                virtual_carries += '1'
                direct_carries += '0'
            else:
                direct_carries += '1'
                virtual_carries += '0'
        else:
            carries += '0'
            direct_carries += '0'
            virtual_carries += '0'
    return carries[::-1], direct_carries[::-1], virtual_carries[::-1]


def get_interleave_sum(a,b):
    interleave = ""
    max_len = max(len(str(a)), len(str(b)))
    for i, (ai, bi) in enumerate(zip(f"{a:0{max_len}d}"[::-1], f"{b:0{max_len}d}"[::-1])):
        interleave += f"{ai}+{bi},"
    return interleave[:-1] # eliminate last ','


def generate_scratchpad(a, b, reverse=True):
    scratchpad = ""
    max_len = max(len(str(a)), len(str(b)))
    for i, (ai, bi) in enumerate(zip(f"{a:0{max_len}d}"[::-1], f"{b:0{max_len}d}"[::-1])):
        ai, bi = int(ai), int(bi)
        sum_ai_bi = f"{ai+bi}"
        op = ',' if i < max_len-1 else ""
        if reverse:
            sum_ai_bi = sum_ai_bi[::-1]
        scratchpad += sum_ai_bi + op
    return scratchpad + ':'


def generate_long_scratchpad(a, b, reverse=True):
    scratchpad = get_interleave_sum(a, b) + '='
    scratch = []
    max_len = max(len(str(a)), len(str(b)))
    for ai, bi in zip(f"{a:0{max_len}d}", f"{b:0{max_len}d}"):
        ai, bi = int(ai), int(bi)
        scratch.append(ai+bi)
    for i in range(1, max_len):
        for j, num in enumerate(scratch[::-1]):
            scratchpad += f"{num:02d}"[::-1] if reverse else f"{num:02d}"
            scratchpad += "," if j < len(scratch)-1 else ':'
        a = scratch.pop()
        b = scratch.pop()
        scratch.append(a+b*10**i)
    return scratchpad


def generate_recursive_scratchpad(a, b, reverse=True):
    scratchpad = ""
    max_len = max(len(str(a)), len(str(b)))
    a_plus_b = str(a+b) 
    a = f"{a:0{max_len}d}"[::-1]
    b = f"{b:0{max_len}d}"[::-1]
    carry = 0
    for i, (ai, bi) in enumerate(zip(a, b)):
        ai, bi = int(ai), int(bi)
        scratchpad += f"{carry}+{ai}+{bi}:"
        scratchpad += str(carry+ai+bi)[::-1] if reverse else str(carry+ai+bi)
        scratchpad += ',:'
        carry = 1 if carry+ai+bi >= 10 else 0
        scratchpad += f"{a_plus_b[::-1][:i+1]}," if reverse else f"{a_plus_b[::-1][:i+1][::-1]},"
        scratchpad += f"{a[i+1:]}+{b[i+1:]}=" if len(a[i+1:]) > 0 else ':'
    return scratchpad

def generate_scratchpad_multiple_addition(numbers: list, reversed_num=True, reversed_order=False, pad_len=0):
    if len(numbers)<1: return ""
    cum_sums = [0]
    for num in numbers[::-1 if reversed_order else 1]:
        cum_sums.append(cum_sums[-1] + num)
    cum_sums_str = ['P'*max(0, pad_len-len(csum)) + csum for csum in map(str, cum_sums[1:])]
    if reversed_num:
        cum_sums_str = [x[::-1] for x in cum_sums_str]
    scratchpad = '>'.join(cum_sums_str)
    return scratchpad

################ END of Helper Functions ################

#########################################################


class AdditionDataset(ArithmeticDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            reverse_input=False,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token=SpecialToken.pad,
            hard_carry=False,
            **kwargs
        ):
        super().__init__()
        self.reverse_input = reverse_input
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        self.pad_token = pad_token
        for _ in trange(n_data):
            if len(self.inputs) >= n_data: break
            if hard_carry:
                n_digits = torch.randint(low=min_n_digits,
                                        high=max_n_digits+1, size=(1,)).item()
                a, b = generate_hard_carry_sum_pair(n_digits)
            else:
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
            result = a + b
            max_len = max(len(str(a)),len(str(b)))

            _inputs = f"{'P'*(max_len-len(str(a)))}{a}+{'P'*(max_len-len(str(b)))}{b}" if padding else f"{a}+{b}"
            _labels = f"{'P'*(max_len+1-len(str(result)))}{result}" if padding else f"{result}"
            if commutative and a != b:
                _inputs = f"{'P'*(max_len-len(str(b)))}{b}+{'P'*(max_len-len(str(a)))}{a}" if padding else f"{b}+{a}"
                _labels = f"{'P'*(max_len+1-len(str(result)))}{result}" if padding else f"{result}"
            self.inputs.append(_inputs)
            self.labels.append(_labels)


#########################################################
    

class AdditionDatasetWithCoupledPositions(AdditionDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            reverse_input=False,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token='0',
            hard_carry=False,
            randomize=True,
            max_position=22,  # max_pos >= len(operand)+2 
            cyclic=False,
            vanilla=False,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        self.cyclic = cyclic
        self.vanilla = vanilla # not a coupled position; vanilla randomized APE
        super().__init__(n_data, min_n_digits, max_n_digits,
            reverse_input, reverse_output, commutative, padding, pad_token,
            hard_carry, **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        a, b = inputs.split('+')
        max_len = max(len(a),len(b))
        start_ = 1 #(self.max_position - (len(labels)+1)) // 2 + 1
        if self.cyclic:
            start = 1 if not self.randomize else torch.randint(1, self.max_position+1, size=(1,)).item()
            end = start + max_len + 1
            positions = list(range(1, self.max_position+1))
            while len(positions) < end:
                positions += list(range(1, self.max_position+1))
            input_positions = positions[end-len(a):end+1] + positions[end-len(b):end+1]
            label_positions = positions[end-len(labels):end]
        elif self.vanilla:
            start = 1 if not self.randomize else torch.randint(1, self.max_position-len(inputs)-len(labels)+1, size=(1,)).item()
            positions = list(range(start, start + len(inputs) + len(labels) + 1))
            input_positions, label_positions = positions[:len(inputs)], positions[len(inputs):]
        else:
            start = start_ if not self.randomize else torch.randint(1, self.max_position-max_len, size=(1,)).item()
            end = start + max_len + 1
            input_positions = list(range(end-len(a), end+1)) + list(range(end-len(b), end+1))
            label_positions = list(range(end-len(labels), end))
        if self.reverse_input:
            inputs = inputs[::-1]
            if not self.vanilla:
                input_positions[:-1] = input_positions[:-1][::-1]
        if self.reverse_output:
            labels = labels[::-1]
            if not self.vanilla:
                label_positions = label_positions[::-1]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))
        labels = " ".join(labels).replace('P', str(self.pad_token))
        return inputs, labels, input_positions, label_positions
    

#########################################################

class AdditionDatasetWithIndexHints(AdditionDataset):
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            reverse_input=False,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token='0',
            hard_carry=False,
            randomize=True,
            max_position=22,  # max_pos >= len(operand)+2 
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        self.padding = padding
        super().__init__(n_data, min_n_digits, max_n_digits,
            reverse_input, reverse_output, commutative, padding, pad_token,
            hard_carry, **kwargs)
        
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        a, b = inputs.split('+')
        a = 'P' + a
        b = 'P' + b
        max_len = max(len(a),len(b))
        start = 10 if not self.randomize else torch.randint(10, self.max_position-max_len+9, size=(1,)).item()
        input_positions_a = list(map(str, range(start, start + len(a))))
        input_positions_b = list(map(str, range(start, start + len(b))))
        label_positions = list(map(str, range(start, start + len(labels))))
        if self.reverse_output:
            labels = labels[::-1]
            label_positions = label_positions[::-1]
        # Interleave positions and tokens
        a = np.array(list(a))
        b = np.array(list(b))
        labels = np.array(list(labels))
        input_positions_a = np.array(input_positions_a)
        input_positions_b = np.array(input_positions_b)
        label_positions = np.array(label_positions)
        a = np.stack([input_positions_a, a]).T.reshape(-1)
        b = np.stack([input_positions_b, b]).T.reshape(-1)
        labels = np.stack([label_positions, labels]).T.reshape(-1)
        # Put white spaces
        a = ' '.join(a.tolist()).replace('P', str(self.pad_token))
        b = ' '.join(b.tolist()).replace('P', str(self.pad_token))
        inputs = a + ' + ' + b
        labels = ' '.join(labels.tolist()).replace('P', str(self.pad_token))
        # Random-starting Position IDs to implement packing-style Position IDs
        inputs_list = inputs.split(' ')
        labels_list = labels.split(' ')
        start = 1 if not self.randomize else torch.randint(1, 6*self.max_position-3-len(inputs_list)-len(labels_list), size=(1,)).item()
        positions = list(range(start, start + len(inputs_list) + len(labels_list) + 1))
        input_positions, label_positions = positions[:len(inputs_list)], positions[len(inputs_list):]
        return inputs, labels, input_positions, label_positions
        

#########################################################


class MultipleAdditionDataset(ArithmeticDataset):
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            min_n_operands=3,
            max_n_operands=3,
            reverse_output=False,
            padding=False,
            pad_token=SpecialToken.pad,
            **kwargs
        ):
        self.min_n_operands = min_n_operands
        self.max_n_operands = max_n_operands
        self.reverse_output = reverse_output
        self.pad_token = pad_token
        self.inputs = []
        self.labels = []
        for i in trange(n_data):
            numbers = []
            # uniform sampling of n_operands 
            n_operands = torch.randint(low=min_n_operands, high=max_n_operands+1, size=(1, )).item()
            # uniform sampling of n_digits of numbers
            n_digits_arr = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(n_operands, ))
            for n_digits in n_digits_arr:
                # uniform sampling of a number
                if n_digits == 1:
                    num = torch.randint(0, 10, size=(1,)).item()
                else:
                    num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                    num = int(''.join(map(str, num_arr)))
                numbers.append(int(num))
            result = sum(numbers)
            max_len = max(len(str(a)) for a in numbers)
            overflow = len(str(int('9'*max_len)*n_operands)) - max_len
            if padding:
                _inputs = '+'.join(f"{'P'*(max_len-len(str(a)))}{a}" for a in numbers)
                _labels = f"{'P'*(max_len+overflow-len(str(result)))}{result}"
            else:
                _inputs = '+'.join(map(str, numbers))
                _labels = str(result)
            self.inputs.append(_inputs)
            self.labels.append(_labels)


#########################################################


class MultipleAdditionDatasetWithCoupledPositions(MultipleAdditionDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            min_n_operands=3,
            max_n_operands=3,
            reverse_output=False,
            padding=False,
            pad_token='0',
            randomize=True,
            max_position=20,
            cyclic=False,
            vanilla=False,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        self.cyclic = cyclic
        self.vanilla = vanilla
        super().__init__(n_data, min_n_digits, max_n_digits,
            min_n_operands, max_n_operands, reverse_output, padding, pad_token, **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        numbers = inputs.split('+')
        max_len = max(map(len, numbers))
        start_ = (self.max_position - (len(labels)+1)) // 2 + 1
        if self.cyclic:
            start = 1 if not self.randomize else torch.randint(1, self.max_position+1, size=(1,)).item()
            end = start + max_len + 1
            positions = list(range(1, self.max_position+1))
            while len(positions) < end:
                positions += list(range(1, self.max_position+1))
            input_positions = sum([positions[end-len(a):end+1] for a in numbers], start=[])
            label_positions = positions[end-len(labels):end]
        elif self.vanilla:
            start = 1 if not self.randomize else torch.randint(1, self.max_position-len(inputs)-len(labels)+1, size=(1,)).item()
            positions = list(range(start, start + len(inputs) + len(labels) + 1))
            input_positions, label_positions = positions[:len(inputs)], positions[len(inputs):]
        else:
            start = start_ if not self.randomize else torch.randint(1, self.max_position-max_len, size=(1,)).item()
            end = start + max_len + 1
            input_positions = sum((list(range(end-len(a), end+1)) for a in numbers), start=[])
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


class MultipleAdditionScratchPadDataset(ArithmeticDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            min_n_operands=2,
            max_n_operands=3,
            reverse_input=True,
            reverse_output=True,
            reverse_output_order=False,
            padding=True,
            pad_token='0',
            **kwargs
        ):
        self.min_n_operands = min_n_operands
        self.max_n_operands = max_n_operands
        self.reverse_input = reverse_input
        self.reverse_output = reverse_output
        self.reverse_output_order = reverse_output_order
        self.pad_token = pad_token
        self.inputs = []
        self.labels = []
        for i in trange(n_data):
            numbers = []
            # uniform sampling of n_operands 
            n_operands = torch.randint(low=min_n_operands, high=max_n_operands+1, size=(1, )).item()
            # uniform sampling of n_digits
            n_digits_arr = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(n_operands, ))
            for n_digits in n_digits_arr:
                # uniform sampling of a number
                if n_digits == 1:
                    num = torch.randint(0, 10, size=(1,)).item()
                else:
                    num_arr = [torch.randint(1, 10, size=(1,)).item()] + torch.randint(0, 10, size=(n_digits-1,)).tolist()
                    num = int(''.join(map(str, num_arr)))
                numbers.append(int(num))
            max_len = max(len(str(a)) for a in numbers)
            overflow = len(str(int('9'*max_len)*n_operands)) - max_len
            result = generate_scratchpad_multiple_addition(
                numbers,
                reversed_num=reverse_output,
                reversed_order=reverse_output_order,
                pad_len=max_len+overflow if padding else 0
            )
            if padding:_inputs = [f"{'P'*(max_len-len(str(a)))}{a}" for a in numbers]
            else: _inputs = list(map(str, numbers))
            if reverse_input: _inputs = [x[::-1] for x in _inputs]            
            _inputs = '+'.join(_inputs)
            self.inputs.append(_inputs)
            self.labels.append(result)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))  # Converts 'P' --> pad_token
        labels = " ".join(labels).replace('P', str(self.pad_token))  # Converts 'P' --> pad_token
        return inputs, labels


#########################################################


class MultipleAdditionScratchPadDatasetWithCoupledPositions(MultipleAdditionScratchPadDataset):                                                                       
    def __init__(self,
            n_data,
            min_n_digits,
            max_n_digits,
            min_n_operands=2,
            max_n_operands=3,
            reverse_input=True,
            reverse_output=True,
            reverse_output_order=False,
            padding=True,
            pad_token='0',
            randomize=True,
            max_position_digits=20,
            max_position_operands=4,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position_digits = max_position_digits
        self.max_position_operands = max_position_operands
        self.padding = padding
        super().__init__(
            n_data,
            min_n_digits,
            max_n_digits,
            min_n_operands,
            max_n_operands,
            reverse_input,
            reverse_output,
            reverse_output_order,
            padding,
            pad_token,
            **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        inp_numbers = inputs.split('+')
        lab_numbers = labels.split('>')
        n_op = len(inp_numbers)
        max_len = max(map(len, inp_numbers + lab_numbers))
        
        start = 1 if not self.randomize else torch.randint(1, self.max_position_digits-max_len, size=(1,)).item()
        input_positions_1 = sum((list(range(start, start+len(a)+1))[::1 if self.reverse_input else -1] for a in inp_numbers), start=[])
        input_positions_1 = input_positions_1[1:] if self.reverse_input else input_positions_1[:-1]
        label_positions_1 = sum((list(range(start, start+len(b)+1))[::1 if self.reverse_output else -1] for b in lab_numbers), start=[]) 
        if not self.reverse_output: label_positions_1 = label_positions_1[-1:] + label_positions_1[:-1]
        
        start = 1 if not self.randomize else torch.randint(1, self.max_position_operands-n_op, size=(1,)).item()
        input_positions_2 = sum(([i] * (len(a)+1) for a, i in zip(inp_numbers, range(start, start+n_op))), start=[])
        input_positions_2 = input_positions_2[1:] if self.reverse_input else input_positions_2[:-1]
        _iter = list(range(start, start+n_op))[::-1 if self.reverse_output_order else 1]
        label_positions_2 = sum(([j] * (len(b)+1) for b, j in zip(lab_numbers, _iter)), start=[])
        
        # Put white spaces
        inputs = " ".join(inputs).replace('P', str(self.pad_token))
        labels = " ".join(labels).replace('P', str(self.pad_token))
        input_positions = [input_positions_1, input_positions_2]
        label_positions = [label_positions_1, label_positions_2]
        return inputs, labels, input_positions, label_positions


#########################################################


class CarryDataset(ArithmeticDataset):
    def __init__(self, 
            n_data, 
            min_n_digits, 
            max_n_digits,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token=SpecialToken.pad,
            mode='all', # all, direct, virtual
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.pad_token = pad_token
        self.inputs = []
        self.labels = []
        for i in range(n_data):
            numbers = []
            if torch.rand(1).item() <= .2:  # 20% of summations are guaranteed to be hard-carry example
                n_digits= torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1, )).item()
                a, b = generate_hard_carry_sum_pair(n_digits)
                numbers = [a, b]
            else:
                # uniform sampling of n_digits of two numbers
                n_digits_arr = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(2, ))
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
                max_len = max(len(str(a)), len(str(b)))
                carries_all, direct_carries, virtual_carries = get_carries_for_addition(a, b)
                if mode == 'all':
                    carries = carries_all
                elif mode == 'direct':
                    carries = direct_carries
                elif mode == 'virtual':
                    carries = virtual_carries
                else: raise ValueError(f'Wrong mode: {mode}')
                self.inputs.append(f"{'P'*(max_len-len(str(a)))}{a}+{'P'*(max_len-len(str(b)))}{b}" 
                                   if padding else f"{a}+{b}")
                self.labels.append(carries)
                if commutative and a != b:
                    self.inputs.append(f"{'P'*(max_len-len(str(b)))}{b}+{'P'*(max_len-len(str(a)))}{a}" 
                                       if padding else f"{b}+{a}")
                    self.labels.append(carries)
    

#########################################################


class CarryDatasetWithCoupledPositions(CarryDataset):
    def __init__(self,
            n_data, 
            min_n_digits, 
            max_n_digits,
            reverse_output=False,
            commutative=False,
            padding=False,
            pad_token='0',
            randomize=True,
            max_position=20,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        super().__init__(n_data, min_n_digits, max_n_digits,
            reverse_output, commutative, padding, pad_token, **kwargs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        a, b = inputs.split('+')
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
        

################################################################
    

class AdditionScratchpad(ArithmeticDataset):
    def __init__(self,
            n_data, 
            min_n_digits, 
            max_n_digits,
            reverse_output=False,
            commutative=False,
            padding=False,
            hard_carry=False,
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        for i in range(n_data):
            if hard_carry:
                n_digits = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(1,)).item()
                numbers = generate_hard_carry_sum_pair(n_digits)
            else:
                numbers = []
                # uniform sampling of n_digits of two numbers
                n_digits_arr = torch.randint(low=min_n_digits, high=max_n_digits+1, size=(2, ))
                for i in range(2):
                    n_digits = n_digits_arr[i].item()
                    # uniform sampling of a number
                    num = torch.randint(low=10**(n_digits-1) - (1 if n_digits==1 else 0),
                                        high=10**n_digits, size=(1,)).item()
                    numbers.append(int(num))
            
            a, b = numbers
            max_len = max(len(str(a)), len(str(b)))
            scratchpad = generate_recursive_scratchpad(a, b, self.reverse_output)
            result = f"{a+b}"
            if reverse_output:
                result = result[::-1]
            self.inputs.append(f"{'P'*(max_len-len(str(a)))}{a}+{'P'*(max_len-len(str(b)))}{b}" 
                                   if padding else f"{a}+{b}")
            self.labels.append(scratchpad+result)
            if commutative and a != b:
                self.inputs.append(f"{'P'*(max_len-len(str(b)))}{b}+{'P'*(max_len-len(str(a)))}{a}" 
                                   if padding else f"{b}+{a}")
                self.labels.append(scratchpad+result)