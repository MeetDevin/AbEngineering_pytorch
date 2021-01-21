# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: main.py
@time: 12/8/20 11:35 AM
@desc:
"""

from data.h5_build import trans_data_to_h5
from data.db_generator_in_torch import construct_dataloader_from_h5
from utils.log_output import *

EXPERIMENT_ID = 'default'

trans_data_to_h5(raw_data_dir='/media/zhangxin/Raid0/dataset/PP', dataset_id='pb_pp2019',
                 force_overwrite=True)
dataloader = construct_dataloader_from_h5(filename='data/preprocessed/pb_pp2019.hdf5', batch_size=5)


for primary, tertiary, label in dataloader:
    print(label)
    pass
pass
