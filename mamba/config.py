import math
import numpy as np
from dataclasses import dataclass
from tensorflow import keras
import tensorflow as tf
from typing import Union

@dataclass
class Config:
    embed_dim: int=64 # Embedding size
    state_dim: int=64 # Number of hidden states in SSM
    expand_ratio: int=2 # hidden_dim = embed_dim * expand * ratio
    kernel_size: int=4 # 1D convolution kernel size before SSM

    # Shape the range of delta(t) inside SSM, Mamba-specified
    min_delta: float=0.001
    max_delta: float=0.1
    delta_scale: float=0.1
    init_floor: float=1e-4

    conv_bias: bool=True # Add a bias term in 1D convolutin
    dense_bias: bool=False # Not add bias in linear projections

    block_id: int=-1 # For tracking use

    max_seq_len: int=128 # Maximum token sequence length for each input
    depth: int=5 # Number of Mamba layers
    dropout_prob: float=0.2 # Dropout rate

    use_decoder: bool=False 
    output_dim: int=None # Number of target classes
    vocab_dim: int=None

    final_act = None # Final layer activation 
    loss_fn: Union[str, keras.losses.Loss]=None # Loss function
    #opt: Union[str, keras.optimizers.Optimizer] = keras.optimizers.legacy.Adam() # Optimizer
    opt: Union[str, tf.keras.optimizers.Optimizer] = tf.keras.optimizers.Adam()

    eval_metrics = ['accuracy'] # Evaluation metrics

    def __post_init__(self):
        self.hidden_dim = int(self.expand_ratio * self.embed_dim)
        self.rank_delta = math.ceil(self.embed_dim / 16) # Dimension of delta projection
        if self.block_id == -1:
            self.block_id = np.round(np.random.randint(0, 1000), 4)
        if self.vocab_dim is None:
            raise ValueError('vocab_dim must be specified')
        if self.use_decoder:
            self.output_dim = self.vocab_dim
        elif self.output_dim is None:
            raise ValueError('output_dim must be specified')
        self.final_act = 'sigmoid' if self.output_dim == 1 else 'softmax'
        if self.loss_fn is None:
            raise ValueError('loss_fn must be defined')



