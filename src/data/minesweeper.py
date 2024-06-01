import numpy as np
from src.data.common import ArithmeticDataset
from tqdm import trange


class MineSweeperGeneratorDataset(ArithmeticDataset):
    def __init__(self, 
            n_data, 
            min_n_len, 
            max_n_len,
            min_mine_ratio,
            max_mine_ratio,
            reverse_output=False,
            **kwargs
        ):
        super().__init__()
        self.reverse_output = reverse_output
        self.inputs = []
        self.labels = []
        self.widths = []
        self.heights = []

        for i in trange(n_data):
            # Randomly sample the size of the board
            n_width = np.random.randint(min_n_len, max_n_len+1)
            n_height = np.random.randint(min_n_len, max_n_len+1)

            # Randomly sample the ratio of the number of mines
            mine_ratio = np.clip(np.random.uniform(min_mine_ratio, max_mine_ratio), 0, 1)

            # Choose the positions of mines as 1, otherwise 0
            random_board = np.random.rand(n_height, n_width)
            one_hot_board = np.where(random_board < mine_ratio, 1, 0)  # 1 == mine, 0 == no mine

            # input board = "mine -> 'M', no mine -> 'E'"
            input_board = np.where(one_hot_board==1, 'M', 'E')

            # Making label board (1): start from a board of size ( (n_height+2) x (n_width+2) )
            label_board = np.zeros((n_height+2, n_width+2), dtype=int)

            # Making label board (2): Fill in!
            label_board[ :n_height  , :n_width  ] += one_hot_board  # upper left
            label_board[ :n_height  ,1:n_width+1] += one_hot_board  # upper 
            label_board[ :n_height  ,2:         ] += one_hot_board  # upper right
            label_board[1:n_height+1, :n_width  ] += one_hot_board  # left
            label_board[1:n_height+1,2:         ] += one_hot_board  # right
            label_board[2:          , :n_width  ] += one_hot_board  # lower left
            label_board[2:          ,1:n_width+1] += one_hot_board  # lower
            label_board[2:          ,2:         ] += one_hot_board  # lower right

            # Making label board (3): take the middle
            label_board = label_board[1:n_height+1, 1:n_width+1]
            label_board = label_board.astype(str)
            label_board[one_hot_board==1] = 'M'  # positions of mines
                        
            self.inputs.append(''.join(map(str,sum(input_board.tolist(), []))))
            self.labels.append(''.join(map(str,sum(label_board.tolist(), []))))
            self.widths.append(n_width)
            self.heights.append(n_height)



class MineSweeperGeneratorDatasetWithCoupledPositions(MineSweeperGeneratorDataset):
    def __init__(self, 
            n_data, 
            min_n_len, 
            max_n_len,
            min_mine_ratio,
            max_mine_ratio,
            reverse_output=False,
            randomize=False,
            max_position=22,
            vanilla=False,
            **kwargs
        ):
        self.randomize = randomize
        self.max_position = max_position
        self.vanilla=vanilla
        super().__init__(n_data, min_n_len, max_n_len, min_mine_ratio, max_mine_ratio,
            reverse_output, **kwargs)
    
    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        n_width = self.widths[index]
        n_height = self.heights[index]

        if self.vanilla:
            start = 1 if not self.randomize else np.random.randint(1, self.max_position-2*n_width*n_height+1)
            input_positions_1 = list(range(start, start+n_width*n_height+1))
            label_positions_1 = list(range(start+n_width*n_height+1, start+2*n_width*n_height+1))
            input_positions_2 = list(range(start, start+n_width*n_height+1))
            label_positions_2 = list(range(start+n_width*n_height+1, start+2*n_width*n_height+1))
        else:
            start_1 = 1 if not self.randomize else np.random.randint(1, self.max_position-n_width+1)
            start_2 = 1 if not self.randomize else np.random.randint(1, self.max_position-n_height+1)

            input_positions_1 = list(range(start_1+1, start_1 + n_width+1, 1)) * n_height + [start_1]
            label_positions_1 = list(range(start_1+1, start_1 + n_width+1, 1)) * n_height
            input_positions_2 = sum([[i] * n_width for i in range(start_2+1, start_2 + n_height+1, 1)], []) + [start_2]
            label_positions_2 = sum([[i] * n_width for i in range(start_2+1, start_2 + n_height+1, 1)], [])

        inputs = " ".join(inputs)
        labels = " ".join(labels)
        input_positions = [input_positions_1, input_positions_2]
        label_positions = [label_positions_1, label_positions_2]
        return inputs, labels, input_positions, label_positions
