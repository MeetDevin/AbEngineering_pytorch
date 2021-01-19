# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: data_generator.py
@time: 12/8/20 8:24 PM
@desc:
"""

import os
import numpy as np
from PIL import Image
from utils.MyException import MyException

class_names = '''Anser+anser
Buteo+buteo
Oriolus+oriolus
Pica+pica'''.split("\n")


class BatchGenerator(object):
    def __init__(self, file_dir, n_classes=8, rate_subset=1, rate_test=0.3, is_one_hot=False,
                 data_format='channels_last'):
        self.file_dir = file_dir
        self.training = True  # 指示当前状态是训练还是测试
        self.epoch_index = 1  # epoch 次数指针，训练从1开始计数，训练数据输送完会指0，开始输送测试数据，next_batch方法会给调用者返回这个值
        self.file_point = 0  # epoch 内的文件指针，每一个新的 epoch 重新归 0

        self.n_classes = n_classes  # 数据集的类别数
        self.rate_subset = rate_subset  # 训练测试所使用的数据，占全部数据的比例
        self.rate_test = rate_test  # 测试数据占使用的数据的比例
        self.is_one_hot = is_one_hot  # 是否使用 one-hot 标签，这里在训练是使用的损失函数是不一样的
        self.data_format = data_format

        self.train_fnames, self.train_labs, self.test_fnames, self.test_labs \
            = self.get_filenames(self.file_dir)

    def get_filenames(self, file_dir):
        """
        遍历全部本地文件名，赋标签；随机打乱顺序；按0.3的比例划分出测试集
        :param file_dir: 文件根目录
        :return: 四个list
        """
        filenames = []
        labels = []

        for train_class in os.listdir(file_dir):
            for pic in os.listdir(file_dir + '/' + train_class):
                if os.path.isfile(file_dir + '/' + train_class + '/' + pic):
                    filenames.append(file_dir + '/' + train_class + '/' + pic)
                    label = class_names.index(train_class)
                    labels.append(int(label))

        filenames_list, lab_list = self.data_to_random(filenames, labels)

        # 子集，当不需要完整数据集的时候，用来生成子集
        n_total = len(filenames_list)
        n_subset = int(n_total * self.rate_subset)
        filenames_list = filenames_list[0:n_subset]
        lab_list = lab_list[0:n_subset]

        # 划分训练和测试
        n_total = len(filenames_list)
        n_test = int(n_total * self.rate_test)
        test_fnames = filenames_list[0:n_test]
        test_labs = lab_list[0:n_test]
        train_fnames = filenames_list[n_test + 1:-1]
        train_labs = lab_list[n_test + 1:-1]

        # labels = [int(i) for i in labels]
        print("训练数据 ：", n_total - n_test)
        print("测试数据 ：", n_test)
        return train_fnames, train_labs, test_fnames, test_labs

    def data_to_random(self, a, b):
        temp = np.array([a, b])
        # 矩阵转置，将数据按行排列，一行一个样本，image位于第一维，label位于第二维
        temp = temp.transpose()
        print('shape of temp:', np.shape(temp))
        # 随机打乱顺序
        np.random.shuffle(temp)
        a_ = list(temp[:, 0])
        b_ = list(temp[:, 1])
        return a_, b_

    def next_batch(self, batch_size, epoch=1):
        """
        获取下一批次（训练或测试）数据
        :param batch_size: 批次大小
        :param epoch: 需要训练的 epoch数，即训练数据集重复训练的遍数
        :return: 数据 numpy数组，标签 numpy数组，和批次索引
        """

        if self.training:
            max = len(self.train_fnames)
        else:
            max = len(self.test_fnames)

        if self.file_point == max or self.file_point + batch_size > max:
            if not self.training:
                # 文件指针到达末尾，并且当前是测试阶段，因此实验完成，抛出异常，让 eager_main.py 接管程序控制
                raise MyException('数据输送完成')
            else:
                # 文件指针到达末尾，当前是训练阶段，因此进入下一个 epoch
                self.epoch_index += 1
                self.file_point = 0
                # 打乱训练数据
                self.train_fnames, self.train_labs = self.data_to_random(self.train_fnames, self.train_labs)
                print('打乱训练数据, epoch=', self.epoch_index)

        if self.epoch_index > epoch:
            # 当完成了 epoch 次重复训练，epoch_index置为0，进入测试阶段
            self.epoch_index = 0  # 第0个epoch表示测试集
            self.file_point = 0
            max = len(self.test_fnames)
            self.training = False
            print('######################### 测试')

        # print('epoch={},point={}'.format(self.epoch_index, self.file_point))

        # 本 batch 的文件结束索引 = 当前文件指针位置 + batch大小
        end = self.file_point + batch_size

        # if end > max:
        # 最后一个批次不足 batch_size 时
        # end = max

        x_data = []  # 训练数据
        y_data = []  # 训练标签，zero-filled list for 'one hot encoding'

        while self.file_point < end and self.file_point < max:
            # 遍历数据，从 file_point 开始，到 end 结束
            if self.training:
                imagePath = self.train_fnames[self.file_point]
            else:
                imagePath = self.test_fnames[self.file_point]
            try:
                # list.shape=[80, 600] 这里可以换成其他任何读取单个样本的数据
                x = np.asarray(Image.open(imagePath))

            # 如果出现数据获取异常，则放弃该数据，获取下一个，为保持 batch_size 恒定， 让 end+1
            except EOFError:
                print('EOFError', imagePath)
                self.file_point += 1
                end += 1
                continue
            except MyException as e:
                # print(e.args)
                self.file_point += 1
                end += 1
                continue

            # 添加颜色通道，为数据增加一个维度
            assert self.data_format == 'channels_first' or self.data_format == 'channels_last'
            if self.data_format == 'channels_first':
                x = np.expand_dims(x, axis=0)
            else:
                x = np.expand_dims(x, axis=-1)

            x_data.append(x)  # (image.data, dtype='float32')

            # 生成标签
            if self.training:
                y_index = int(self.train_labs[self.file_point])
            else:
                y_index = int(self.test_labs[self.file_point])

            if self.is_one_hot:
                y_true = np.zeros(int(self.n_classes), dtype=np.int32)
                y_true[y_index] = 1

            else:
                y_true = y_index

            y_data.append(y_true)

            # 文件指针自增，获取下一个文件
            self.file_point += 1

        # print(np.shape(np.asarray(x_data, dtype=np.float32)))
        return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.int32), self.epoch_index

