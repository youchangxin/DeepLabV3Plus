# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import cv2

from config import cfg

rgb_mean = cfg.RGB_MEAN
rgb_std  = cfg.RGB_STD


class Dataset(object):
    def __init__(self, dataset_type):
        self.image_txt  = cfg.TRAIN.IMG_TXT if dataset_type == 'train' else cfg.TEST.IMG_TXT
        self.labels_dir = cfg.TRAIN.LABEL_DIR if dataset_type == 'train' else cfg.TEST.LABEL_DIR
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug   = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else False

        self.image_label_paths = self._create_image_label_path(self.image_txt, self.labels_dir)
        self.num_samples = self.count_sample()
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def count_sample(self):
        num = os.listdir(self.labels_dir)
        return len(num)

    def _process_image_label(self, image_path, label_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # data augmentation
        # randomly shift gamma
        if self.data_aug:
            gamma = random.uniform(0.8, 1.2)
            image = image.copy() ** gamma
            image = np.clip(image, 0, 255)
            # randomly shift brightness
            brightness = random.uniform(0.5, 2.0)
            image = image.copy() * brightness
            image = np.clip(image, 0, 255)
        # image normalization
        image = (image / 255. - rgb_mean) / rgb_std
        label = open(label_path).readlines()
        label = [np.array(line.rstrip().split(" ")) for line in label]
        label = np.array(label, dtype=np.int)
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int)

        return image, label

    def _create_image_label_path(self, images_filepath, labels_filepath):
        image_paths = open(images_filepath).readlines()
        all_label_txts = os.listdir(labels_filepath)
        image_label_paths = []
        for label_txt in all_label_txts:
            label_name = label_txt[:-4]
            label_path = labels_filepath + "/" + label_txt
            for image_path in image_paths:
                image_path = image_path.rstrip()
                image_name = image_path.split("/")[-1][:-4]
                if label_name == image_name:
                    image_label_paths.append((image_path, label_path))

        random.shuffle(image_label_paths)
        return image_label_paths

    def __iter__(self):
        return self

    def __next__(self):
        """
        generate image and mask at the same time
        """
        images = np.zeros(shape=[self.batch_size, 512, 512, 3])
        labels = np.zeros(shape=[self.batch_size, 512, 512], dtype=np.float)
        if self.batch_count < self.num_batchs:
            for i in range(self.batch_size):
                index = self.batch_count * self.batch_size + i
                if index >= self.num_samples:
                    index -= self.num_samples

                image_path, label_path = self.image_label_paths[index]
                image, label = self._process_image_label(image_path, label_path)
                images[i], labels[i] = image, label
            self.batch_count += 1
            return images, labels
        else:
            self.batch_count = 0
            raise StopIteration

    def __len__(self):
        return self.num_batchs




