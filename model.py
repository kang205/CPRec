import tensorflow as tf


class Model():
    def __init__(self, usernum, itemnum, args):
        self.usernum = usernum
        self.itemnum = itemnum

        d_emb = args.latent_dimension
        learning_rate = args.learning_rate

        self.u_embeddings = tf.Variable(tf.random_uniform([usernum, d_emb], maxval=0.1))
        self.i_embeddings = tf.Variable(tf.random_uniform([itemnum, d_emb], maxval=0.1))
        self.Xc = tf.Variable(
            tf.random_normal([d_emb, d_emb], stddev=1 / (d_emb ** 0.5), dtype=tf.float32)
        )
        self.Xp = tf.Variable(
            tf.random_normal([d_emb, d_emb], stddev=1 / (d_emb ** 0.5), dtype=tf.float32)
        )

        self.item_bias = tf.Variable(tf.zeros([itemnum]))

        self.batch_u = tf.placeholder(tf.int32, [None])
        self.batch_i = tf.placeholder(tf.int32, [None])
        self.batch_j = tf.placeholder(tf.int32, [None])
        self.batch_oi = tf.placeholder(tf.int32, [None])
        self.batch_oj = tf.placeholder(tf.int32, [None])

        self._batch_u_emb = tf.gather(self.u_embeddings, self.batch_u)
        self.batch_u_emb = tf.matmul(self._batch_u_emb, self.Xc)
        self.batch_i_emb = tf.gather(self.i_embeddings, self.batch_i)
        self.batch_j_emb = tf.gather(self.i_embeddings, self.batch_j)
        self._batch_oi_emb = tf.gather(self.u_embeddings, self.batch_oi)
        self.batch_oi_emb = tf.matmul(self._batch_oi_emb, self.Xp)
        self._batch_oj_emb = tf.gather(self.u_embeddings, self.batch_oj)
        self.batch_oj_emb = tf.matmul(self._batch_oj_emb, self.Xp)
        self.batch_i_bias = tf.gather(self.item_bias, self.batch_i)
        self.batch_j_bias = tf.gather(self.item_bias, self.batch_j)

        pos_distances = self.batch_i_bias + tf.reduce_sum(self.batch_u_emb * self.batch_i_emb, 1) + tf.reduce_sum(
            self.batch_u_emb * self.batch_oi_emb, 1)
        neg_distances = self.batch_j_bias + tf.reduce_sum(self.batch_u_emb * self.batch_j_emb, 1) + tf.reduce_sum(
            self.batch_u_emb * self.batch_oj_emb, 1)
        self.rank_loss = -tf.reduce_sum(tf.log_sigmoid(pos_distances - neg_distances))

        self.reg_loss = sum(map(tf.nn.l2_loss,
                                [self._batch_u_emb, self.batch_i_emb, self.batch_j_emb, self._batch_oi_emb,
                                 self._batch_oj_emb, self.batch_i_bias, self.batch_j_bias]
                                )
                            ) * args.lambda1 + \
                        sum(map(tf.nn.l2_loss, [self.Xc, self.Xp])) * args.batch_size * args.lambda2

        self.loss = self.rank_loss + self.reg_loss

        self.auc = tf.reduce_mean((tf.sign(-neg_distances + pos_distances) + 1) / 2)

        self.gds = []
        self.gds.append(tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(self.loss))
