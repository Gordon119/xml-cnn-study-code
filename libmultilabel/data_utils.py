from collections import *
import logging
import os
import re

import torch
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile

from libmultilabel.utils import pad_sequence

UNK = "<unk>"
PAD = "<pad>"

GLOVE_WORD_EMBEDDING = {
    "glove.42B.300d",
    "glove.840B.300d",
    "glove.6B.50d",
    "glove.6B.100d",
    "glove.6B.200d",
    "glove.6B.300d",
}

class TextDataset(Dataset):
    """Class for text dataset"""

    def __init__(self, data, word_dict, classes, max_seq_length):
        self.data = data
        self.word_dict = word_dict.word_dict
        self.classes = classes
        self.max_seq_length = max_seq_length
        self.num_classes = len(self.classes)
        self.label_binarizer = MultiLabelBinarizer().fit([classes])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return {
            'text': torch.LongTensor([self.word_dict.get(word, self.word_dict[UNK]) for word in data['text']][:self.max_seq_length]),
            'label': torch.IntTensor(self.label_binarizer.transform([data['label']])[0]),
        }


def generate_batch(data_batch, max_len=None):
    text_list = [data['text'] for data in data_batch]
    label_list = [data['label'] for data in data_batch]
    return {
        'text': pad_sequence(text_list, batch_first=True, max_len=max_len),
        'label': torch.stack(label_list)
    }


def get_dataset_loader(
    data,
    word_dict,
    classes,
    device,
    max_seq_length=500,
    batch_size=1,
    shuffle=False,
    fixed_length=False,
    data_workers=4
):
    dataset = TextDataset(data, word_dict, classes, max_seq_length)

    if fixed_length:
        collate_fn = lambda batch: generate_batch(batch, max_seq_length)
    else:
        collate_fn = generate_batch

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_workers,
        collate_fn=collate_fn,
        pin_memory='cuda' in device.type,
    )
    return dataset_loader


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]


def _load_raw_data(path, is_test=False):
    logging.info(f'Load data from {path}.')
    data = pd.read_csv(path, sep='\t', names=['label', 'text'],
                       converters={'label': lambda s: s.split(),
                                   'text': tokenize})
    data = data.reset_index().to_dict('records')
    if not is_test:
        data = [d for d in data if len(d['label']) > 0]
    return data


def load_datasets(
    data_dir,
    train_path=None,
    test_path=None,
    val_path=None,
    val_size=0.2,
    is_eval=False
):
    datasets = {}
    test_path = test_path or os.path.join(data_dir, 'test.txt')
    if is_eval:
        datasets['test'] = _load_raw_data(test_path, is_test=True)
    else:
        if os.path.exists(test_path):
            datasets['test'] = _load_raw_data(test_path, is_test=True)
        train_path = train_path or os.path.join(data_dir, 'train.txt')
        datasets['train'] = _load_raw_data(train_path)
        val_path = val_path or os.path.join(data_dir, 'valid.txt')
        if os.path.exists(val_path):
            datasets['val'] = _load_raw_data(val_path)
        elif val_size > 0:
            datasets['train'], datasets['val'] = train_test_split(
                datasets['train'], test_size=val_size, random_state=42)
        else:
            datasets['val'] = datasets['test']

    msg = ' / '.join(f'{k}: {len(v)}' for k, v in datasets.items())
    logging.info(f'Finish loading dataset ({msg})')
    return datasets

class AttributeDict(dict):
    """AttributeDict is an extended dict that can access
    stored items as attributes.

    >>> ad = AttributeDict({'ans': 42})
    >>> ad.ans
    >>> 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_used", set())

    def __getattr__(self, key: str) -> any:
        try:
            value = self[key]
            self._used.add(key)
            return value
        except KeyError:
            raise AttributeError(f'Missing attribute "{key}"')

    def __setattr__(self, key: str, value: any) -> None:
        self[key] = value
        self._used.discard(key)

    def used_items(self) -> dict:
        """Return the items that have been used at least once after being set.

        Returns:
            dict: The used items.
        """
        return {k: self[k] for k in self._used}

def load_or_build_text_dict(
    dataset,
    vocab_file=None,
    min_vocab_freq=1,
    embed_file=None,
    embed_cache_dir=None,
    silent=False,
):
    """Build or load the vocabulary from the training dataset or the predefined `vocab_file`.
    The pretrained embedding can be either from a self-defined `embed_file` or from one of
    the vectors: `glove.6B.50d`, `glove.6B.100d`, `glove.6B.200d`, `glove.6B.300d`, `glove.42B.300d`, or `glove.840B.300d`.

    Args:
        dataset (list): List of training instances with index, label, and tokenized text.
        vocab_file (str, optional): Path to a file holding vocabuaries. Defaults to None.
        min_vocab_freq (int, optional): The minimum frequency needed to include a token in the vocabulary. Defaults to 1.
        embed_file (str): Path to a file holding pre-trained embeddings or the name of the pretrained GloVe embedding. Defaults to None.
        embed_cache_dir (str, optional): Path to a directory for storing cached embeddings. Defaults to None.
        silent (bool, optional): Enable silent mode. Defaults to False.
        normalize_embed (bool, optional): Whether the embeddings of each word is normalized to a unit vector. Defaults to False.

    Returns:
        tuple[dict, torch.Tensor]: A dictionary which maps tokens to indices and the pre-trained word vectors of shape (vocab_size, embed_dim).
    """
    if vocab_file:
        logging.info(f"Load vocab from {vocab_file}")
        with open(vocab_file, "r") as fp:
            vocab_list = [[vocab.strip() for vocab in fp.readlines()]]
        # Keep PAD index 0 to align `padding_idx` of
        # class Embedding in libmultilabel.nn.networks.modules.
        word_dict = _build_word_dict(vocab_list, min_vocab_freq=1, specials=[PAD, UNK])
    else:
        vocab_list = [set(data["text"]) for data in dataset]
        word_dict = _build_word_dict(vocab_list, min_vocab_freq=min_vocab_freq, specials=[PAD, UNK])

    logging.info(f"Read {len(word_dict)} vocabularies.")

    embedding_weights = get_embedding_weights_from_file(word_dict, embed_file, silent, embed_cache_dir)

    return AttributeDict({"word_dict": word_dict, "vectors":embedding_weights})


def _build_word_dict(vocab_list, min_vocab_freq=1, specials=None):
    r"""Build word dictionary, modified from `torchtext.vocab.build-vocab-from-iterator`
    (https://docs.pytorch.org/text/stable/vocab.html#build-vocab-from-iterator)

    Args:
        vocab_list: List of words.
        min_vocab_freq (int, optional): The minimum frequency needed to include a token in the vocabulary. Defaults to 1.
        specials: Special tokens (e.g., <unk>, <pad>) to add. Defaults to None.

    Returns:
        dict: A dictionary which maps tokens to indices.
    """

    counter = Counter()
    for tokens in vocab_list:
        counter.update(tokens)

    # sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # add special tokens at the beginning
    tokens = specials or []
    for token, freq in ordered_dict.items():
        if freq >= min_vocab_freq:
            tokens.append(token)

    # build token to indices dict
    word_dict = dict()
    for idx, token in enumerate(tokens):
        word_dict[token] = idx
    return word_dict

def load_or_build_label(datasets, label_file=None, silent=False):
    if label_file:
        logging.info('Load labels from {label_file}')
        with open(label_file, 'r') as fp:
            classes = sorted([s.strip() for s in fp.readlines()])
    else:
        classes = set()
        for dataset in datasets.values():
            for d in tqdm(dataset, disable=silent):
                classes.update(d['label'])
        classes = sorted(classes)
    return classes


def get_embedding_weights_from_file(word_dict, embed_file, silent=False, cache_dir=None):
    """Obtain the word embeddings from file. If the word exists in the embedding file,
    load the pretrained word embedding. Otherwise, assign a zero vector to that word.
    If the given `embed_file` is the name of a pretrained GloVe embedding, the function
    will first download the corresponding file.

    Args:
        word_dict (dict): A dictionary for mapping tokens to indices.
        embed_file (str): Path to a file holding pre-trained embeddings or the name of the pretrained GloVe embedding.
        silent (bool, optional): Enable silent mode. Defaults to False.
        cache_dir (str, optional): Path to a directory for storing cached embeddings. Defaults to None.

    Returns:
        torch.Tensor: Embedding weights (vocab_size, embed_size).
    """

    if embed_file in GLOVE_WORD_EMBEDDING:
        embed_file = _download_glove_embedding(embed_file, cache_dir=cache_dir)
    elif not os.path.isfile(embed_file):
        raise ValueError(
            "Got embed_file {}, but allowed pretrained " "embeddings are {}".format(embed_file, GLOVE_WORD_EMBEDDING)
        )

    logging.info(f"Load pretrained embedding from {embed_file}.")
    with open(embed_file) as f:
        word_vectors = f.readlines()
    embed_size = len(word_vectors[0].split()) - 1

    vector_dict = {}
    for word_vector in tqdm(word_vectors, disable=silent):
        word, vector = word_vector.rstrip().split(" ", 1)
        vector = torch.Tensor(list(map(float, vector.split())))
        vector_dict[word] = vector

    embedding_weights = torch.zeros(len(word_dict), embed_size)
    # Add UNK embedding
    #   AttentionXML: np.random.uniform(-1.0, 1.0, embed_size)
    #   CAML: np.random.randn(embed_size)
    unk_vector = torch.randn(embed_size)
    embedding_weights[word_dict[UNK]] = unk_vector

    # Store pretrained word embedding
    vec_counts = 0
    for word in word_dict.keys():
        if word in vector_dict:
            embedding_weights[word_dict[word]] = vector_dict[word]
            vec_counts += 1

    logging.info(f"Loaded {vec_counts}/{len(word_dict)} word embeddings")

    return embedding_weights

def _download_glove_embedding(embed_name, cache_dir=None):
    """Download pretrained glove embedding from https://huggingface.co/stanfordnlp/glove/tree/main.

    Args:
        embed_name (str): The name of the pretrained GloVe embedding. Defaults to None.
        cache_dir (str, optional): Path to a directory for storing cached embeddings. Defaults to None.

    Returns:
        str: Path to the file that contains the cached embeddings.
    """
    cache_dir = ".vector_cache" if cache_dir is None else cache_dir
    cached_embed_file = f"{cache_dir}/{embed_name}.txt"
    if os.path.isfile(cached_embed_file):
        return cached_embed_file
    os.makedirs(cache_dir, exist_ok=True)

    remote_embed_file = re.sub(r"6B.*", "6B", embed_name) + ".zip"
    url = f"https://huggingface.co/stanfordnlp/glove/resolve/main/{remote_embed_file}"
    logging.info(f"Downloading pretrained embeddings from {url}.")
    try:
        zip_file, _ = urlretrieve(url, f"{cache_dir}/{remote_embed_file}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(cache_dir)
    except Exception as e:
        os.remove(zip_file)
        raise e
    logging.info(f"Downloaded pretrained embeddings {embed_name} to {cached_embed_file}.")
    return cached_embed_file
