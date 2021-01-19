# encoding: utf-8
"""
@author: Xin Zhang
@contact: zhangxin@szbl.ac.cn
@file: MyException.py
@time: 12/9/20 11:57 AM
@desc:
"""


class MyException(Exception):
    """
    继承自基类 Exception
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message
