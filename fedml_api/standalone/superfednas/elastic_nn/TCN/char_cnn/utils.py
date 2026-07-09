import os
import json
import numpy as np
import torch

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def process_x(raw_x_batch, device):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        x_batch = torch.from_numpy(x_batch)
        x_batch = x_batch.to(device)
        return x_batch

def process_y(raw_y_batch, device):
    y_batch = [word_to_indices(word) for word in raw_y_batch]
    y_batch = torch.tensor(y_batch)
    y_batch = y_batch.to(device)
    return y_batch

def batch_data(data, batch_size, seed, device):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']
    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    batches = []
    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = process_x(data_x[i:i+batch_size], device)
        batched_y = process_y(data_y[i:i+batch_size], device)
        batches.append((batched_x, batched_y))
    return batches

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    #Big error in fedml codebase if (user, group) pairings are important
    #Appears as thought group information is irrelevant outside of there being same number of groups as users?
    #multiple levels of useless
    #clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

#doesn't support different client number and different partition methods beside non. iid.
#Need to generate a different json dataset using LEAF github to get different non. iid. or iid. distributions: https://github.com/TalwalkarLab/leaf

def load_shakespeare(data_dir, batch_size, seed, device):
    #Tracking information
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()

    train_path = data_dir + "/train"
    test_path = data_dir+ "/test"
    users, groups, train_data, test_data = read_data(train_path, test_path)
    if len(groups) == 0:
        groups = [None for _ in users]
    client_num = 0

    sentence_len = None

    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])
        user_test_data_num = len(test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_num] = user_train_data_num

        #Train Data
        for ind, (sentence_slice, next_letter) in enumerate(zip(train_data[u]['x'], train_data[u]['y'])):
            if sentence_len is None:
                sentence_len = len(sentence_slice)
            else:
                assert sentence_len == len(sentence_slice), "inconsistent sentence length in train dataset!"
            train_data[u]['y'][ind] = (sentence_slice[1:]+next_letter)

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size, seed, device)

        #update dict and global dataset
        train_data_local_dict[client_num] = train_batch
        train_data_global += train_batch

        #Test Data
        for ind, (sentence_slice, next_letter) in enumerate(zip(test_data[u]['x'], test_data[u]['y'])):
            if sentence_len is None:
                sentence_len = len(sentence_slice)
            else:
                assert sentence_len == len(sentence_slice), "inconsistent sentence length in test dataset!"
            test_data[u]['y'][ind] = (sentence_slice[1:]+next_letter)

        # transform to batches
        test_batch = batch_data(test_data[u], batch_size, seed, device)

        #update dict and global dataset
        test_data_local_dict[client_num] = test_batch
        test_data_global += test_batch

        client_num += 1

    output_dim = NUM_LETTERS

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim
