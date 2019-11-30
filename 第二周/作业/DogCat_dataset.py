# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @author     : yts3221@126.com
# @date       : 2019-08-21 10:08:00
# @brief      : 各数据集的Dataset定义
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
DogCat_label = {"dog": 0, "cat": 1}


class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        # print(self.data_info) #[('./test_data\\dog\\dog.9.jpg', 0)]
        self.transform = transform #读取图片处理

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):#绝对路径 路径下文件夹 绝对路径下文件
            # print(root, dirs, _)
            # 遍历类别
            for sub_dir in dirs:
                # print(sub_dir)
                img_names = os.listdir(os.path.join(root, sub_dir))#遍历所有文件夹下的图片进行路径拼接找出图片
                # print(img_names)
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names)) #过滤jpg结尾的数据
                # print(img_names)

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)#拼接图片路径
                    label = DogCat_label[sub_dir]#找出label 根据文件夹名
                    data_info.append((path_img, int(label)))
        # print(data_info)   #[('./test_data\\cat\\dog.9.jpg', 1), ('./test_data\\dog\\dog.10.jpg', 0)]

        return data_info
