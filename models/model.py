import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, args):
        self.is_training = args.is_training
        self.is_testing = args.is_testing
        self.is_predict = args.is_predict
        self.is_return_embedding = args.is_return_embedding
        self.is_return_udid_embedding = args.is_return_udid_embedding
        self.embedding_size = args.embedding_size
        self.gameid_count = args.gameid_count
        self.model_count = args.model_count
        self.datatraceid_count = args.datatraceid_count
        self.build_model()


    def build_model(self):
        #placeholder
        self.hist_click = tf.placeholder(tf.int32, [None, None])  #历史点击的gameid
        self.sl = tf.placeholder(tf.int32, [None, ])              #历史点击序列的长度
        self.last_click = tf.placeholder(tf.int32, [None, ])      #最后点击的gameid
        self.model = tf.placeholder(tf.int32, [None, ])           #model
        #self.basic = tf.placeholder(tf.float32, [None, 4])       #基础特征
        self.sub_sample = tf.placeholder(tf.int32, [None, None])  #[正样本, 负样本]
        self.y = tf.placeholder(tf.float32, [None, None])         #[正样本, 负样本]对应的标签[1, 0]
        self.datatrace = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32)               #防止过拟合
        self.lr = tf.placeholder(tf.float64)                      #学习率


        #embedding
        gameid_emb_w = tf.get_variable("gameid_emb_w", [self.gameid_count, self.embedding_size])
        model_emb_w = tf.get_variable("model_emb_w", [self.model_count, self.embedding_size])
        datatraceid_emb_w = tf.get_variable("datatraceid_emb_w", [self.datatraceid_count, self.embedding_size])

        #gameid的偏置
        input_b = tf.get_variable("input_b", [self.gameid_count], initializer=tf.constant_initializer(0.0))

        #获取历史点击序列的embedding
        h_emb = tf.nn.embedding_lookup(gameid_emb_w, self.hist_click)          #shape(batch, sl, embedding_size)
        hist = tf.reduce_mean(h_emb, 1)
        # 获取历史datatraceid的embedding
        d_emb = tf.nn.embedding_lookup(datatraceid_emb_w, self.datatrace)  # shape(batch, sl, embedding_size)
        d_emb = tf.reduce_mean(d_emb, 1)

        # mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32) #shape(batch, sl)
        # mask =tf.expand_dims(mask, -1)                                         #shape(batch, sl, 1)
        # mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])                       #shape(batch, sl, embedding_size)
        # h_emb *= mask                                                          #shape(batch, sl, embedding_size)
        # hist = tf.reduce_sum(h_emb, 1)                                         #shape(batch, embedding_size)
        # #池化(历史点击embedding取均值)
        # hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, self.embedding_size]), tf.float32)) #shape(batch, embedding_size)


        #获取最后一次点击embedding
        last_emb = tf.nn.embedding_lookup(gameid_emb_w, self.last_click)       #shape(batch, embedding_size)
        #获取model embedding
        model_emb = tf.nn.embedding_lookup(model_emb_w, self.model)            #shape(batch, embedding_size)

        #self.input = tf.concat([hist, last_emb, model_emb, self.basic], axis=-1)
        self.input = tf.concat([hist, d_emb, last_emb, model_emb], axis=-1)           #shape(batch, 3*embedding_size)

        bn = tf.layers.batch_normalization(inputs=self.input, name="b1")
        layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.relu, name='f1')
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.layers.dense(layer_1, 512, activation=tf.nn.relu, name='f2')
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)
        layer_3 = tf.layers.dense(layer_2, self.embedding_size, activation=tf.nn.relu, name="f3")  #shape(batch, embedding_size)

        #softmax
        if self.is_training:
            sample_b = tf.nn.embedding_lookup(input_b, self.sub_sample)
            sample_w = tf.concat([tf.nn.embedding_lookup(gameid_emb_w, self.sub_sample),
                                 #tf.tile(tf.expand_dims(self.basic, 1), [1, tf.shape(sample_b)[1], 1])
                                 ], axis=2)                                    #shape(batch, sample, embedding_size)

            #用户向量矩阵
            user_v = tf.expand_dims(layer_3, 1)                                #shape(batch, 1, embedding_size)
            sample_w = tf.transpose(sample_w, perm=[0, 2, 1])                  #shape(batch, embedding_size, sample)

            self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b

            #Step variable
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.yhat = tf.nn.softmax(self.logits)
            self.loss = tf.reduce_mean(-self.y * tf.log(self.yhat + 1e-24))
            tf.summary.scalar("loss", self.loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        if self.is_testing or self.is_predict:
            self.logits = tf.matmul(layer_3, gameid_emb_w, transpose_b=True) + input_b   #shape(batch, gameid_count)
            self.output = tf.nn.softmax(self.logits)                                     #shape(batch, gameid_count)

        if self.is_return_embedding:
            self.gameid_emb_w = gameid_emb_w
            self.model_emb_w = model_emb_w

        if self.is_return_udid_embedding:
            self.udid_emb_w = layer_3


    def train(self, sess, uij, lr, summary_op, keep_prob):
        loss, summmary, step, _ = sess.run([self.loss, summary_op, self.global_step, self.train_op], feed_dict = {
            self.hist_click: uij[1],  # 历史点击的gameid
            self.sl: uij[2],          # 历史点击序列的长度
            self.model: uij[3],       # model
            self.datatrace: uij[4],   #datatrace
            self.last_click: uij[5],  # 最后点击的gameid
            self.sub_sample: uij[6],  # [正样本, 负样本]
            self.y: uij[7],           # [正样本, 负样本]对应的标签[1, 0]
            # self.basic: uij[8],       # 基础特征
            self.keep_prob: keep_prob,# 防止过拟合
            self.lr: lr,              # 学习率
        })

        return loss, summmary, step, _


    def test(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={

            self.hist_click: uij[1],     #历史点击的gameid
            self.sl: uij[2],             #历史点击序列长度
            self.model: uij[3],          #model
            self.datatrace: uij[4],      #datatraceid
            self.last_click: uij[5],     #最后点击的gameid
            # self.basic: uij[7],          #基础特征
            self.keep_prob: keep_prob    #防止过拟合
        })

    ##输出embedding矩阵
    def embedding(self, sess):
        return sess.run([self.gameid_emb_w, self.model_emb_w])


    ##输出用户embedding矩阵
    def embedding_of_user(self, sess, uij, keep_prob):
        return sess.run(self.udid_emb_w, feed_dict = {
            self.hist_click: uij[1],   # 历史点击的gameid
            self.sl: uij[2],           # 历史点击序列长度
            self.model: uij[3],        # model
            self.datatrace: uij[4],    # datatraceid
            self.last_click: uij[5],   # 最后点击的gameid
            # self.basic: uij[7],        # 基础特征
            self.keep_prob: keep_prob  # 防止过拟合
        })


    def save(self, sess, path, global_step):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path, global_step=global_step)

    def load(self, sess, path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, save_path=ckpt.model_checkpoint_path)
            print("Load model of step %s success" % step)
        else:
            print("No checkpoint!")