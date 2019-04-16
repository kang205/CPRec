import tensorflow as tf
import numpy as np
from sampler import WarpSampler
from model import Model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--latent_dimension', default=20, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--maximum_epochs', default=2000, type=int)
# need tuning for new data, the default value is for reddit
# best parameter for Pinerest data is also 0.1
parser.add_argument('--lambda1', default=0.1, type=float)
# need tuning for new data, the default value is for reddit
# best parameter for Pinerest data is 0.001
parser.add_argument('--lambda2', default=0.0001, type=float)
args = parser.parse_args()

dataset = np.load('data/' + args.dataset + 'Partitioned.npy')

[user_train, user_validation, user_test, usernum, itemnum] = dataset

f = open(
    'CPRec_%s_%d_%g_%g_%g.txt' % (args.dataset, args.latent_dimension, args.learning_rate, args.lambda1, args.lambda2),
    'w')

# count positive events
owner = dict()
oneiteration = 0
for user in range(usernum):
    oneiteration += len(user_train[user]['consume'])
    for item in set(user_train[user]['produce']):
        if item in owner:
            print "multiple_creators!"
        owner[item] = user
for item in range(itemnum):
    if item not in owner:
        print "missing creator!"
        break
oneiteration = min(1000000, oneiteration)

sampler = WarpSampler(user_train, user_validation, user_test, owner, usernum, itemnum, batch_size=args.batch_size,
                      n_workers=1)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

best_valid_auc = 0.5
best_iter = 0

for i in range(args.maximum_epochs):
    for _ in range(oneiteration / args.batch_size):
        batch = sampler.next_train_batch()
        batch_u, batch_i, batch_j, batch_oi, batch_oj = batch
        _, train_loss, train_auc = sess.run((model.gds, model.loss, model.auc),
                                            {model.batch_u: batch_u,
                                             model.batch_i: batch_i,
                                             model.batch_j: batch_j,
                                             model.batch_oi: batch_oi,
                                             model.batch_oj: batch_oj
                                             })
        print train_loss, train_auc

    if i % 10 == 0:
        f.write('#iter %d: loss %f, train auc %f \n' % (i, train_loss, train_auc))
        _valid_auc = 0.0
        _test_auc = 0.0
        n_batch = 1000000 / args.batch_size
        for _ in range(n_batch):
            batch = sampler.next_valid_batch()
            batch_u, batch_i, batch_j, batch_oi, batch_oj = batch

            valid_loss, valid_auc = sess.run((model.loss, model.auc),
                                             {model.batch_u: batch_u,
                                              model.batch_i: batch_i,
                                              model.batch_j: batch_j,
                                              model.batch_oi: batch_oi,
                                              model.batch_oj: batch_oj})

            batch = sampler.next_test_batch()
            batch_u, batch_i, batch_j, batch_oi, batch_oj = batch
            test_loss, test_auc = sess.run((model.loss, model.auc),
                                           {model.batch_u: batch_u,
                                            model.batch_i: batch_i,
                                            model.batch_j: batch_j,
                                            model.batch_oi: batch_oi,
                                            model.batch_oj: batch_oj})
            _valid_auc += valid_auc
            _test_auc += test_auc

        _valid_auc /= n_batch
        _test_auc /= n_batch
        f.write('%f %f\n' % (_valid_auc, _test_auc))
        f.flush()
        if _valid_auc > best_valid_auc:
            best_valid_auc = _valid_auc
            best_test_auc = _test_auc
            best_iter = i
        elif i >= best_iter + 50:
            break
f.write('Finished! %f, %f\n' % (best_valid_auc, best_test_auc))
f.close()
