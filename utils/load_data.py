# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

rx_indexes_of_manytx = [
    '1-1', '1-19', '1-20', '13-7',
    '14-7', '18-19', '18-2', '19-1',
    '19-2', '2-1', '20-1', '20-19',
    '3-19', '7-14', '7-7', '8-14',
    '8-7', '8-8']

MIN_SAMPLES = 50


def preprocessing(x: np.ndarray) -> np.ndarray:
    for i in range(x.shape[0]):
        power = np.sum(x[i, 0, :] ** 2 + x[i, 1, :] ** 2) / x.shape[2]
        x[i] = x[i] / np.sqrt(power)
    return x


def load_single_dataset(
    dataset: str,
    rx_index: int,
    date_index: int,
    tx_num: int,
    is_eq: str = 'non_equalized',
    dataset_root: str = "dataset",
):
    if dataset == 'ManyTx':
        rx_indexes = rx_indexes_of_manytx
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    folder_path = os.path.join(dataset_root, dataset, is_eq)
    file_path = os.path.join(folder_path, f'date{date_index}',
                             f'rx_{rx_indexes[rx_index]}_data.pkl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at path: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    x_list, y_list, dropped = [], [], []
    for tx_index in range(tx_num):
        tx_data = data['data'][tx_index]
        tx_data_formatted = np.transpose(tx_data, (0, 2, 1))
        if tx_data_formatted.shape[0] < MIN_SAMPLES:
            dropped.append((tx_index, tx_data_formatted.shape[0]))
            continue
        tx_data_formatted = tx_data_formatted[:MIN_SAMPLES]
        x_list.append(tx_data_formatted)
        y_list.extend([tx_index] * MIN_SAMPLES)

    if not x_list:
        return np.empty((0, 2, tx_data_formatted.shape[-1]), np.float32), np.array([], np.int64)

    x = preprocessing(np.concatenate(x_list, axis=0).astype(np.float32))
    y = np.array(y_list, dtype=np.int64)
    return x, y