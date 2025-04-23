import tensorflow as tf
from tensorflow.keras import layers, Model
from residual_block import HighwayBlock
from transformers import AutoTokenizer

def build_model(config, tokenizer_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    inp = layers.Input(shape=(config.max_seq_len,), name='token_ids')
    emb = layers.Embedding(config.vocab_dim, config.embed_dim, input_length=config.max_seq_len)(inp)

    x = emb
    for i in range(config.depth):
        x = HighwayBlock(config, name=f'Block_{i}')(x)
        x = layers.Dropout(config.dropout_prob)(x)

    
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    if not config.use_decoder:
        x = layers.Flatten()(x)
    x = layers.Dense(1024, activation=tf.nn.gelu)(x) # GELU better than ReLU
    logits = layers.Dense(config.output_dim, activation=config.final_act)(x)

    model = Model(inputs=inp, outputs=logits, name='SelectiveSeq')
    model.compile(loss=config.loss_fn, optimizer=config.opt, metrics=config.eval_metrics)

    return model, tokenizer