import tensorflow as tf

def predict(text, model, tokenizer, config):

    # Need to Modify!

    # Works for AutoTokenizer, including .encode()
    # inputs = tokenizer.encode(text, max_length=config.max_seq_len, padding='max_length', return_tensors='tf')

    # Works for DummyTokenizer, like run_test example
    tokenized = tokenizer([text])['input_ids']
    inputs = tf.convert_to_tensor(tokenized)


    return model(inputs)[-1, 0]