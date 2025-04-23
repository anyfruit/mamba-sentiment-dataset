from tensorflow.keras import layers
from mamba_block import SelectiveBlock

class HighwayBlock(layers.Layer):
    # Should include:
    # Layer normalization (mean of 0, std of 1)
    # Mamba Block
    # Residual connection

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = SelectiveBlock(config)
        self.normalizer = layers.LayerNormalization(epsilon=1e-5) 


    def call(self, x):
        # HighwayBlock(x) = SelectiveBlock(LayerNorm(x)) + x
        return self.processor(self.normalizer(x)) + x  
        # Borrowed from Transformer and ResNet, residual paths help gradients flow