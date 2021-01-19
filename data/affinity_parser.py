# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: affinity_parser.py
@time: 1/19/21 5:21 PM
@desc:
"""


def get_affinity(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()[6:]  # start by line 7
        lines = [line.split() for line in lines]  # divide up by space key
        id_affinity = [[line[0], line[3]] for line in lines]  # get pdb id and affinity
        dic = dict(id_affinity)
    return dic
