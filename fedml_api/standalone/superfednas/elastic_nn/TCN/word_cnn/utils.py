import os
import torch
from torch.autograd import Variable
import pickle

"""
Note: The meaning of batch_size in word_cnn is different from that in MNIST example. In MNIST, 
batch_size is the # of sample data that is considered in each iteration; in word_cnn, however,
it is the number of segments to speed up computation. 
The goal of word_cnn is to train a language model to predict the next word.
"""
# option to force re-make corpus not supported (check https://github.com/alindkhare/TCN/blob/alind/elastic_tcn/TCN/word_cnn/word_cnn_test.py)
# eval batch size is fixed to 10


def data_generator(data_dir):
    corpus = pickle.load(open(data_dir + '/corpus', 'rb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, '../../../../../../data/PTB/train.txt'))
        self.valid = self.tokenize(os.path.join(path, '../../../../../../data/PTB/valid.txt'))
        self.test = self.tokenize(os.path.join(path, '../../../../../../data/PTB/test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, batch_size, device):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    data = data.to(device)
    return data


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i+seq_len], requires_grad=False)
    target = Variable(source[:, i+1:i+1+seq_len], requires_grad=False)     # CAUTION: This is un-flattened!
    return data, target


def load_ptb(data_dir, partition_method, client_num_in_total, batch_size, device):
    if partition_method != "homo":
        raise Exception(f"Unsupported partition method: {partition_method}!")
    corpus = data_generator(data_dir)
    eval_batch_size = 10
    train_data_local_num_dict = []
    train_data_local_dict = []
    test_data_local_dict = []
    n_train = corpus.train.shape[0]
    client_train_size = int(n_train/client_num_in_total)
    for i in range(client_num_in_total):
        train_data_local_dict.append(batchify(corpus.train[(i*client_train_size):((i+1)*client_train_size)], batch_size, device))
        train_data_local_num_dict.append(client_train_size)
        test_data_local_dict.append(batchify(corpus.test, eval_batch_size, device))
    train_data = batchify(corpus.train, batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    return corpus.train.shape[0], corpus.test.shape[0], train_data, test_data, train_data_local_num_dict,\
           train_data_local_dict, test_data_local_dict, len(corpus.dictionary)
