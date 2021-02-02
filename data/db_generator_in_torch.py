# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: data_generator_in_torch.py
@time: 12/8/20 8:12 PM
@desc:
"""

import h5py
import torch
from utils.log_output import write_out
from torch.utils.data import DataLoader


def construct_dataloader_from_h5(filename, batch_size):
    dataset = H5PytorchDataset(filename)
    write_out('construct dataloader... total: ', dataset.__len__(), ', ', batch_size, ' per batch.')
    return torch.utils.data.DataLoader(H5PytorchDataset(filename),
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2)


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5py_file = h5py.File(filename, 'r')
        self.num_samples, self.max_sequence_len = self.h5py_file['primary'].shape

    def __getitem__(self, index):
        primary = torch.Tensor(self.h5py_file['primary'][index])
        tertiary = torch.Tensor(self.h5py_file['tertiary'][index])  # max length x 9
        label = torch.Tensor(self.h5py_file['label'][index])
        return primary, tertiary, label

    def __len__(self):
        return self.num_samples

#
# def merge_samples_to_batch(batch):
#     # samples_list = []
#     # for sample in batch:
#     #     samples_list.append(sample)
#     # # sort according to length of aa sequence
#     # samples_list.sort(key=lambda x: len(x[0]), reverse=True)
#     prim = []
#     teri = []
#     lab = []
#     for p, s, l in batch:
#         prim.append(torch.Tensor(p))
#         teri.append(torch.Tensor(s))
#         lab.append(torch.Tensor(l))
#
#     return prim, teri, lab
