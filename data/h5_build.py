# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: h5_build.py
@time: 12/9/20 9:48 AM
@desc: Inspired by repository https://github.com/biolib/openprotein
"""
import glob
import os.path
import os
import numpy as np
import h5py
import platform
import sys
import torch

from utils.process_bar import show_process_realtime
from utils.log_output import write_out
from data.pdb_parser import get_primary_tertiary
from data.affinity_parser import get_affinity
from utils.MyException import MyException

MAX_SEQUENCE_LEN = 5000  # 氨基酸主链的最大队列长度, 即限定一个蛋白质最多有2,000个 amino acid
MAX_DATASET_LEN = 10000  # max length of data, consider this value for your PC memory


def trans_data_to_h5(raw_data_dir="data/raw/", dataset_id='default', padding=True, force_overwrite=True):
    H5Founder(raw_data_dir=raw_data_dir, dataset_id=dataset_id, padding=padding, force_overwrite=force_overwrite)


class H5Founder:
    def __init__(self, raw_data_dir="data/raw/", dataset_id='default', padding=False, force_overwrite=True):
        self.raw_data_dir = raw_data_dir
        self.dataset_id = dataset_id
        self.padding = padding
        self.force_overwrite = force_overwrite

        self.h5_name = "data/preprocessed/" + self.dataset_id + ".hdf5"
        self.filenames, self.affinity_dic = self.traversal_PBDbind2019()

        # 如果 .hdf5 文件已经存在，选择强制写入还是跳过
        if os.path.isfile(self.h5_name):
            write_out("Preprocessed file for " + self.dataset_id + " already exists.")
            if self.force_overwrite:
                write_out("force_pre_processing_overwrite flag set to True, "
                          "overwriting old file...")
                os.remove(self.h5_name)
                self.construct_h5db_from_list()
            else:
                write_out("Skipping pre-processing...")
        else:
            self.construct_h5db_from_list()
        write_out("Completed pre-processing.")

    def traversal_PBDbind2019(self):
        """
        Traversal raw data in data/raw/，and use function process_file() to read data in loop at the same time
        output preprocessed data in the format .hdf5 in data/preprocessed/
        """
        write_out("Starting pre-processing of raw data...")
        # glob 查找符合特定规则的文件 full path.
        # 匹配符："*", "?", "[]"。"*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，[0-9]匹配数字。
        files_list = glob.glob(self.raw_data_dir + '/*')
        files_filtered_list = filter_input_files(files_list)  # list['filename', ...]

        affinity_dic = get_affinity(file_path=self.raw_data_dir + '/index/INDEX_general_PP.2019')
        return files_filtered_list, affinity_dic

    def construct_h5db_from_list(self):
        """
        这个函数是在 process_raw_data 的循环里调用的，每一个文件会调用一次
        :param input_file: data/raw\\protein_net_testfile.txt
        :param h5_name: data/preprocessed/" + filename + ".hdf5
        """
        # create output file
        file = h5py.File(self.h5_name, 'w')
        current_buffer_size = 1
        file_point = 0
        num_files = len(self.filenames)

        dataset_p = file.create_dataset(name='primary', shape=(current_buffer_size, MAX_SEQUENCE_LEN),
                                        maxshape=(MAX_DATASET_LEN, MAX_SEQUENCE_LEN),
                                        dtype='int')  # amino acid sequences
        dataset_s = file.create_dataset(name='tertiary', shape=(current_buffer_size, MAX_SEQUENCE_LEN, 9),
                                        maxshape=(MAX_DATASET_LEN, MAX_SEQUENCE_LEN, 9), dtype='float')  # structures
        dataset_l = file.create_dataset(name='label', shape=(current_buffer_size, 1),
                                        maxshape=(MAX_DATASET_LEN, 1), dtype='float')

        for file_path in self.filenames:
            # write_out("Writing ", file_path)
            if platform.system() == 'Windows':
                pdb_id = file_path.split('\\')[-1].replace('.ent.pdb', '')
            else:
                pdb_id = file_path.split('/')[-1].replace('.ent.pdb', '')

            try:
                primary, tertiary, length = get_primary_tertiary(file_path, pdb_id=pdb_id)  # get structure
                affinity = self.affinity_dic.get(pdb_id)  # get affinity
            except MyException:
                write_out('> MyException occurred, skip this file: '+file_path)
                num_files -= 1
                continue

            if length > MAX_SEQUENCE_LEN:
                # 跳过长度超过MAX_SEQUENCE_LEN的蛋白质
                write_out("Dropping protein as length too long:", length)
                num_files -= 1
                continue

            # 追加空间分配, current_buffer_size 应该是样本数量, 并且指向栈顶
            if file_point >= current_buffer_size:
                current_buffer_size = current_buffer_size + 1
                dataset_p.resize((current_buffer_size, MAX_SEQUENCE_LEN))
                dataset_s.resize((current_buffer_size, MAX_SEQUENCE_LEN, 9))
                dataset_l.resize((current_buffer_size, 1))

            if self.padding:
                primary_padded = np.zeros(MAX_SEQUENCE_LEN)
                tertiary_padded = np.zeros((MAX_SEQUENCE_LEN, 9))

                primary_padded[:length] = primary
                tertiary_padded[:length, :] = tertiary
            else:
                primary_padded = primary
                tertiary_padded = tertiary

            dataset_p[file_point] = primary_padded
            dataset_s[file_point] = tertiary_padded
            dataset_l[file_point] = affinity

            show_process_realtime(file_point, num_files, name='trans h5')
            file_point += 1


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    allowed_file_endings = ".pdb"
    rate = 0.03
    partical_input_files = input_files[:int(len(input_files)*rate)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(allowed_file_endings), partical_input_files))
