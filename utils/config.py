import numpy as np

# config
diff = 1
image_height, image_width, image_channel = 192, 64, 3
NumCAPTCHA, NumAlb = 4, 36
batchsize = 50

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

def input_index_generate_batch(image_batch, label_batch):
    def get_input_lens(sequences):
        # 64 is the output channels of the last layer of CNN
        lengths = np.asarray([64 for _ in sequences], dtype=np.int64)
        return sequences, lengths
    batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
    batch_labels = sparse_tuple_from_label(label_batch)
    return batch_inputs, batch_seq_len, batch_labels