import numpy as np


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        data_batch = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        # udid, click_list, datatraceid_list, last_model, last_click, 正样本(label), 负样本(20个)
        u, model, datatrace, items, y, his_len, last_click, basic = [], [], [], [], [], [], [], []
        for t in data_batch:
            u.append(t[0])                              # udid
            his_len.append(len(t[1]))                   # histroy click seq (历史点击的gameid个数)
            model.append(t[3])                          # model
            last_click.append(t[4])                     # last click item  (最后一次点击的item)
            items.append([t[5]] + t[6])                 # 正样本 + 负样本
            sub_sample_size = len(t[6]) + 1             # 样本总数
            mask = np.zeros(sub_sample_size, np.int64)  # 正负样本标签
            mask[0] = 1                                 # 第一个为正样本
            y.append(mask)                              # label
            #basic.append(t[7])                         #基础特征(后续可加)

        max_sl = max(his_len)                           # 每个batch历史点击个数最多的那条样本
        hist_i = np.zeros([len(data_batch), max_sl], np.int64)      # shape(batch_size, max_sl)
        datatrace = np.zeros([len(data_batch), max_sl], np.int64)   # shape(batch_size, max_sl)
        for k, t in enumerate(data_batch):
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]                              # 依次填充历史点击的gameid
                datatrace[k][l] = t[2][l]                           # 依次填充历史点击的datatraceid

        return self.i, (u, hist_i, his_len, model, datatrace, last_click, items, y, basic)

class DataInputTest:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        data_batch = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        # udid, click_list, datatraceid_list, last_model, last_click, 正样本(label)
        u, model, datatrace, his_len, last_click, label, basic = [], [], [], [], [], [], []
        for t in data_batch:
            u.append(t[0])             # udid
            his_len.append(len(t[1]))  # histroy click seq length
            model.append(t[3])         # model
            last_click.append(t[4])    # 最后一个点击
            label.append(t[5])         # 标签gameid
            #basic.append(t[6])

        max_sl = max(his_len)

        hist_i = np.zeros([len(data_batch), max_sl], np.int64)
        datatrace = np.zeros([len(data_batch), max_sl], np.int64)  # shape(batch_size, max_sl)
        for k, t in enumerate(data_batch):
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                datatrace[k][l] = t[2][l]

        return self.i, (u, hist_i, his_len, model, datatrace, last_click, label, basic)


class DataInputPred:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        data_batch = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        # udid, click_list, datatraceid_list, last_model, last_click
        u, model, datatrace, his_len, last_click, basic = [], [], [], [], [], []
        for t in data_batch:
            u.append(t[0])             # udid
            his_len.append(len(t[1]))  # histroy click seq length
            model.append(t[3])         # model
            last_click.append(t[4])    # 最后一个点击
            #basic.append(t[5])

        max_sl = max(his_len)

        hist_i = np.zeros([len(data_batch), max_sl], np.int64)
        datatrace = np.zeros([len(data_batch), max_sl], np.int64)  # shape(batch_size, max_sl)
        for k, t in enumerate(data_batch):
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                datatrace[k][l] = t[2][l]

        return self.i, (u, hist_i, his_len, model, datatrace, last_click, basic)


