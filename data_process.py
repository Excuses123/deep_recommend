import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import gc

def build_map(df, col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])

    return m, key

def main(path_in, path_out, neg_size=50):
    if  path_in.split(".")[-1] == "csv":
        data = pd.read_csv(path_in)
    else:
        data = pd.read_parquet(path_in)
    data = data.sort_values(by=['timestamp'], ascending=True).drop(columns=['datekey', 'testid', 'isdownload', 'isclick'], axis=1)

    gameid_map, gameid_key = build_map(data, "gameid")
    model_map, model_key = build_map(data, "model")
    datatraceid_map, datatraceid_key = build_map(data, "datatraceid")
    udid_map, udid_key = build_map(data, 'udid')

    udid_count, gameid_count, model_count, datatraceid_count, data_count = len(udid_key), len(gameid_key), len(model_key), len(datatraceid_key), len(data)
    print('udid_count: %d\tgameid_count: %d\tmodel_count: %d\tdatatraceid_count: %d\tdata_count: %d' % (udid_count, gameid_count, model_count, datatraceid_count, data_count))

    train_set = []  #训练用
    test_set = []   #测试用
    pred_set = []   #预测用(包含所有用户)
    ###test udid = 9826
    for udid, hist in tqdm(data.groupby('udid')):
        pos_list = hist['gameid'].tolist()
        len_pos = len(pos_list)
        model_list = hist['model'].tolist()# 取last_click对应的model
        datatraceid_list = hist['datatraceid'].tolist()# 历史下载的datatrace
        # udid, click_list, datatraceid_list, last_model, last_click
        pred_set.append((udid, pos_list, datatraceid_list, model_list[-1], pos_list[-1]))
        # 用户点击记录大于1个时，最近的点击拿去测试
        if len_pos > 1:
            # udid, click_list, datatraceid_list, last_model, last_click, 正样本(label)
            test_set.append((udid, pos_list[:-1], datatraceid_list[:-1], model_list[-2], pos_list[-2], pos_list[-1]))
        # 用户点击记录大于2个时，参与训练
        if len_pos > 2:
            def gen_neg():
                neg = pos_list[0]
                while neg in pos_list:  # 判断是否点击过
                    neg = np.random.randint(0, gameid_count - 1)
                return neg

            # 随机生成300个该用户未点过的游戏作为负样本
            neg_list = [gen_neg() for i in range(neg_size * len_pos)]
            neg_list = np.array(neg_list)

            for i in range(max(1, len_pos-15), len_pos - 1):
                # 从300个负样本候选集中随机选取20个负样本
                index = np.random.randint(len(neg_list), size=neg_size)
                # udid, click_list, datatraceid_list, last_model, last_click, 正样本(label), 负样本(20个)
                train_set.append((udid, pos_list[:i], datatraceid_list[:i], model_list[i - 1], pos_list[i - 1], pos_list[i], list(neg_list[index])))

    '''
    (2525,
      2520,
      [1098, 123],
      123,
      1168,
      [1340,128,36,2893,81,2363,81,2893,1110,2454,1340,3016,2893,1110,3016,2486,1123,1354,86,1256])
    '''
    del data
    gc.collect()

    print("train set first row: ", train_set[0])
    print('train len: ', len(train_set))
    print('test len: ', len(test_set))
    print('pred len: ', len(pred_set))

    with open(path_out + "dataset_train.pkl", 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((udid_count, gameid_count, model_count, datatraceid_count), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((gameid_key, model_key, datatraceid_key, udid_key), f, pickle.HIGHEST_PROTOCOL)

    with open(path_out + "dataset_test.pkl", 'wb') as f:
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((udid_count, gameid_count, model_count, datatraceid_count), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((gameid_key, model_key, datatraceid_key, udid_key), f, pickle.HIGHEST_PROTOCOL)

    with open(path_out + "dataset_pred.pkl", 'wb') as f:
        pickle.dump(pred_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((udid_count, gameid_count, model_count, datatraceid_count), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((gameid_key, model_key, datatraceid_key, udid_key), f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    data_path = "./data/train_click.csv"
    out_path = "./data/dataset.pkl"
    main(data_path, out_path)



