# mnist_dataloader.py

from partition_data_dirichlet import split_index
import logging
import params
import numpy as np
import torch
from MNISTReader import MNISTImageReader, MNISTLabelReader

def batch_data(data, batch_size):
    data_x = data["x"]
    data_y = data["y"]
    # randomly shuffle data
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = torch.tensor(data_x[i : i + batch_size], dtype=torch.float32)
        batched_y = torch.tensor(data_y[i : i + batch_size], dtype=torch.long)
        batch_data.append((batched_x, batched_y))
    return batch_data

def read_data(args):
    train_images_path = params.MNIST_TRAIN_IMAGES_PATH
    train_labels_path = params.MNIST_TRAIN_LABELS_PATH
    test_images_path = params.MNIST_EVAL_IMAGES_PATH
    test_labels_path = params.MNIST_EVAL_LABELS_PATH

    with MNISTImageReader(train_images_path) as img_reader:
        _, train_images = img_reader.read(params.DATASET_SIZE_USED_TO_TRAIN)

    with MNISTLabelReader(train_labels_path) as lbl_reader:
        _, train_labels = lbl_reader.read(params.DATASET_SIZE_USED_TO_TRAIN)

    with MNISTImageReader(test_images_path) as img_reader:
        _, test_images = img_reader.read(params.DATASET_SIZE_USED_TO_EVAL)

    with MNISTLabelReader(test_labels_path) as lbl_reader:
        _, test_labels = lbl_reader.read(params.DATASET_SIZE_USED_TO_EVAL)

    client_idcs, df_client = split_index(train_labels, args.client_num_in_total, 'non-iid', args.partition_alpha)
    eclient_idcs, edf_client = split_index(test_labels, args.client_num_in_total, 'non-iid', args.partition_alpha)

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    for i in range(args.client_num_in_total):
        clients.append('f_0000{}'.format(i))
        groups.append(None)

        index_client_i = np.argwhere(df_client == i).reshape(-1)
        label_i = train_labels[index_client_i]
        data_i = train_images[index_client_i]
        train_data['f_0000{}'.format(i)] = {'y': label_i.tolist(),
                                            'x': data_i.tolist()}

        eindex_client_i = np.argwhere(edf_client == i).reshape(-1)
        elabel_i = test_labels[eindex_client_i]
        edata_i = test_images[eindex_client_i]
        test_data['f_0000{}'.format(i)] = {'y': elabel_i.tolist(),
                                           'x': edata_i.tolist()}

    return clients, groups, train_data, test_data

def load_partition_data_mnist(args, batch_size):
    users, groups, train_data, test_data = read_data(args)
    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num
        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)
        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10
    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )

def load_data(args):
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
    )
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num

if __name__ == '__main__':
    pass



