from __future__ import unicode_literals
from functools import reduce
import tensorflow as tf
import numpy as np
import warnings
import argparse
import skimage.io
import skimage.transform
import skimage
import scipy.io.wavfile
from multiprocessing import Process, Queue


class SequenceData():
    def __init__(self, path, batch_size=32):
        self.path = path
        self.batch_size = batch_size
        f = open(path)
        self.datas = f.readlines()
        self.L = len(self.datas)
        self.index = random.sample(range(self.L), self.L)
        self.queue = Queue(maxsize=30)

        self.Process_num = 32
        for i in range(self.Process_num):
            print(i, 'start')
            ii = int(self.__len__() / self.Process_num)
            t = Process(target=self.f, args=(i * ii, (i + 1) * ii))
            t.start()

    def __len__(self):
        return self.L - self.batch_size

    def __getitem__(self, idx):
        batch_indexs = self.index[idx:(idx + self.batch_size)]
        batch_datas = [self.datas[k] for k in batch_indexs]
        img1s, img2s, audios, labels = self.data_generation(batch_datas)
        return img1s, img2s, audios, labels

    def f(self, i_l, i_h):
        for i in range(i_l, i_h):
            t = self.__getitem__(i)
            self.queue.put(t)

    def gen(self):
        while 1:
            t = self.queue.get()
            yield t[0], t[1], t[2], t[3]

    def data_generation(self, batch_datas):
        # 数据预处理操作
        return img1s, img2s, audios, labels


epochs = 2

data_g = SequenceData('train_1.csv', batch_size=48)
dataset = tf.data.Dataset().batch(1).from_generator(data_g.gen,
                                                    output_types=(tf.float32, tf.float32, tf.float32, tf.float32))
X, y, z, w = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        for j in range(int(len(data_g) / (data_g.batch_size))):
            face1, face2, voice, labels = sess.run([X, y, z, w])
            print(face1.shape)
