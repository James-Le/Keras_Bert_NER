class PaddingInputExample(object):
    
class InputExample(object):
    
    def __init__(self, guid, text_a, text_b=None, labels=None):
        
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=50):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        labels = [tags["-PAD-"]] * max_seq_length
        return input_ids, input_mask, segment_ids, labels

    new_labels = copy.deepcopy(example.labels)
    tokens_a = tokenizer.tokenize(example.text_a)
    
    for idx, t in enumerate(tokens_a):
        try:
            dummy = new_labels[idx]
        except IndexError as e:
            new_labels.insert(idx, new_labels[idx-1])
        if t[:2] == "##":
            new_labels.insert(idx, new_labels[idx-1])        
        
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]
        new_labels = new_labels[0 : (max_seq_length - 2)]
        
    tokens = []
    segment_ids = []
    labels = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    labels.append(tag2idx["-PAD-"])
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        labels.append(new_labels[i])
    labels.append(tag2idx["-PAD-"])
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    input_mask = [1] * len(input_ids)
    
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        
    while len(labels) < max_seq_length:
        labels.append(tag2idx["-PAD-"])
        
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(labels) == max_seq_length
    
    return input_ids, input_mask, segment_ids, labels

def convert_examples_to_features(tokenizer, examples, max_seq_length=50):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels_arr, shapetags_arr = [], [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, labels = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels_arr.append(labels)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array([to_categorical(i, num_classes=n_tags) for i in labels_arr]),
    )

def convert_text_to_examples(texts, labels_arr):
    """Create InputExamples"""
    InputExamples = []
    for text, labels in zip(texts, labels_arr):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None,
                         labels=" ".join(labels))
        )
    return InputExamples

