
import numpy as np
import pandas as pd
import os, sys, time, pickle
import tensorflow as tf
from models.model import Model
from models.input import DataInput, DataInputTest, DataInputPred, DataSequence
import data_process

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


def main():
    # init args
    args = Args()
    if not os.path.exists(args.cache_path):
        print("generate cache data!")
        data_process.main(args.data_path, args.cache_path, args.neg_size)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    with open(args.cache_path, 'rb') as f:
        args.train_set = pickle.load(f)
        args.test_set = pickle.load(f)
        args.pred_set = pickle.load(f)
        args.udid_count, args.gameid_count, args.model_count = pickle.load(f)
        args.gameid_key, args.model_key, args.udid_key = pickle.load(f)
    print("udid_count: %d\tgameid_count: %d\tmodel_count: %d" % (args.udid_count, args.gameid_count, args.model_count))
    print("train set size: ", len(args.train_set))
    # build model
    model = Model(args)

    return args, model


def get_batch_data(train_data, epoch_num, batch_size):
    train_data = [str(i) for i in train_data]
    # 从tensor列表中按顺序或随机抽取一个tensor准备放入文件名称队列
    input_queue = tf.train.slice_input_producer([train_data], num_epochs=epoch_num, shuffle=True)
    # 从文件名称队列中读取文件准备放入文件队列
    batch_data = tf.train.batch(input_queue, batch_size=batch_size, num_threads=4, capacity=64, allow_smaller_final_batch=False)

    return batch_data



def train(args, model, epoch, lr, batch_size):
    print("training.............")
    # batch_data = get_batch_data(args.train_set, epoch, batch_size)

    with tf.Session(config=gpu_config) as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 强制刷新缓冲区
        sys.stdout.flush()
        if args.is_training:
            epoch_size = round(len(args.train_set) / batch_size)
            print("every_epoch_step: ", epoch_size)

            summary_op = tf.summary.merge_all()                                    # 合并summary并初始化所有变量
            train_writer = tf.summary.FileWriter(args.checkpoint_dir, sess.graph)  # 保存模型静态图
            coord = tf.train.Coordinator()                                         # 开启一个协调器
            threads = tf.train.start_queue_runners(sess, coord)                    # 启动队列填充
            loss_sum = 0.0
            start_time = time.time()
            try:
                while not coord.should_stop():
                    loss, summmary, step, _ = model.train(sess, batch_data, lr, summary_op, keep_prob=0.8)
                    train_writer.add_summary(summmary, step)
                    loss_sum += loss
                    # 每迭代100次打印损失信息
                    if step % 100 == 0 or step == 0:
                        print('Global_step:%d\tTrain_loss:%.4f' % (model.global_step.eval(), loss_sum / step))
                    # 每1000步保存模型
                    if step % 5000 == 0:
                        model.save(sess, args.checkpoint_dir + 'model.ckpt', global_step=step)
                        # print('Global_step:%d\tTrain_loss:%.4f' % (model.global_step.eval(), loss_sum / model.global_step.eval()))
            except tf.errors.OutOfRangeError:
                print("done! now lets kill all the threads……")
            finally:
                #协调器发出所有线程终止信号
                coord.request_stop()
                print('all threads are asked to stop!')
            coord.join(threads)
            print("all threads are stopped!")
            end_time = time.time()
            print("train model use time: %d sec" % (end_time - start_time))
            # 保存最终模型
            model.save(sess, args.checkpoint_dir + 'model.ckpt', global_step=model.global_step.eval())

            if args.is_return_embedding:
                gameid_emb_w, model_emb_w = model.embedding(sess)
                pd.DataFrame(gameid_emb_w).to_csv("./output/train_gameid_emb.csv")
                pd.DataFrame(model_emb_w).to_csv("./output/train_model_emb.csv")
        else:
            print("no train")

def test(args, model, batch_size, out_file="test_result.txt"):
    print("test..............")
    with tf.Session(config=gpu_config) as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 强制刷新缓冲区
        sys.stdout.flush()
        if args.is_testing:
            model.load(sess, args.checkpoint_dir)  # + 'model.ckpt') #+ '-1000')  #加载指定迭代次数的模型
            output_file = open(args.output_path + out_file, "w")
            output_file.write("\t".join(['udid','hist_click','label','pred_list']) + "\n")
            # uij: (u, model, hist_i, sl, lt, lb); len(uij[0])=batch  shape(8, batch)
            for _, uij in DataInputTest(args.test_set, batch_size):
                output = model.test(sess, uij, keep_prob=1.0)  # shape(batch, item_count)
                pred_index = np.argsort(-output, axis=1)[:, 0:20]  # shape(batch, 20) (索引)
                # 遍历一个batch_size的udid
                for i in range(len(uij[0])):
                    # udid
                    output_file.write(str(args.udid_key[uij[0][i]]) + "\t")
                    # hist_click
                    output_file.write(",".join([str(args.gameid_key[i]) for i in uij[2][i] if i != 0]) + "\t")
                    # label
                    output_file.write("%i\t" % args.gameid_key[uij[5][i]])
                    # pred
                    pred = np.array(args.gameid_key)[pred_index[i]].astype(str)
                    output_file.write(",".join(pred) + "\n")
            output_file.close()
            if args.is_return_embedding:
                gameid_emb_w, model_emb_w = model.embedding(sess)
                pd.DataFrame(gameid_emb_w).to_csv("./output/test_gameid_emb.csv")
                pd.DataFrame(model_emb_w).to_csv("./output/test_model_emb.csv")
        else:
            print("no test")

def pred(args, model, batch_size, out_file="predict_result.txt"):
    print("predict...........")
    with tf.Session(config=gpu_config) as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 强制刷新缓冲区
        sys.stdout.flush()
        if args.is_predict:
            model.load(sess, args.checkpoint_dir)  # + 'model.ckpt') #+ '-1000')  #加载指定迭代次数的模型
            output_file = open(args.output_path + out_file, "w")
            output_file.write("udid\tpred_list\n")
            # uij: (u, model, hist_i, sl, lt); len(uij[0])=batch  shape(8, batch)
            for _, uij in DataInputPred(args.pred_set, batch_size):
                output = model.test(sess, uij, keep_prob=1.0)  # shape(batch, item_count)
                pred_index = np.argsort(-output, axis=1)[:, 0:200]  # shape(batch, 20) (索引)
                # 遍历一个batch_size的udid
                for i in range(len(uij[0])):
                    # udid
                    output_file.write(str(args.udid_key[uij[0][i]]) + "\t")
                    # pred
                    pred = np.array(args.gameid_key)[pred_index[i]].astype(str)
                    output_file.write(",".join(pred) + "\n")
            output_file.close()
        else:
            print("no pred")


class Args():
    data_path = "./data/train_click.csv"
    cache_path = "./data/dataset.pkl"
    checkpoint_dir = './ckpt/'
    output_path = "./output/"
    embedding_size = 256
    neg_size = 50
    is_training = True
    is_testing = False
    is_predict = False
    is_return_embedding = False
    is_return_udid_embedding = False
    #训练数据
    train_set = []
    test_set = []
    pred_set = []
    udid_count = []
    gameid_count = -1
    model_count = -1
    gameid_key = []
    model_key = []
    udid_key = []


if __name__ == '__main__':
    epoch = 30
    lr = 0.001
    train_batch_size = 48
    test_batch_size = 100

    args, model = main()
    train(args, model, epoch, lr, train_batch_size)
    test(args, model, test_batch_size)
    #计算召回
    pred(args, model, test_batch_size)



















