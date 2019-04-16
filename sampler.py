import numpy as np
import threading
import Queue


def sample_function(user_train, user_validation, user_test, owner, usernum, itemnum, batch_size, train_queue,
                    valid_queue, test_queue, SEED):
    def sample_ui():
        if not is_test:
            user = np.random.randint(0, usernum)
            while len(User[user]['consume']) == 0: user = np.random.randint(0, usernum)
        else:
            user = np.random.randint(0, usernum)
            while len(User_test[user]['consume']) == 0: user = np.random.randint(0, usernum)
        num_item = len(User[user]['consume'])

        # find postive item pair
        if not is_test:
            item_i = np.random.randint(0, num_item)
            item_i = User[user]['consume'][item_i]
        else:
            item_i = User_test[user]['consume'][0]

        # find negtive item
        item_ip = np.random.randint(0, itemnum)
        while item_ip in User[user]['consume'] or item_ip == item_i: item_ip = np.random.randint(0, itemnum)
        return user, item_i, item_ip, owner[item_i], owner[item_ip]

    np.random.seed(SEED)
    User = user_train
    while True:
        if not train_queue.full():
            is_test = False
            User_test = []
            one_batch = []
            for i in range(batch_size):
                batch = sample_ui()
                one_batch.append(batch)
            train_queue.put_nowait(zip(*one_batch))
        if not valid_queue.full():
            is_test = True
            User_test = user_validation
            one_batch = []
            for i in range(batch_size):
                batch = sample_ui()
                one_batch.append(batch)
            valid_queue.put_nowait(zip(*one_batch))
        if not test_queue.full():
            is_test = True
            User_test = user_test
            one_batch = []
            for i in range(batch_size):
                batch = sample_ui()
                one_batch.append(batch)
            test_queue.put_nowait(zip(*one_batch))


class WarpSampler(object):

    def __init__(self, user_train, user_validation, user_test, owner, usernum, itemnum, batch_size=10000, n_workers=2):
        self.train_queue = Queue.Queue(maxsize=n_workers)
        self.valid_queue = Queue.Queue(maxsize=n_workers)
        self.test_queue = Queue.Queue(maxsize=n_workers)
        self.threads = []

        for i in range(n_workers):
            self.threads.append(threading.Thread(target=sample_function, args=(user_train,
                                                                               user_validation,
                                                                               user_test,
                                                                               owner,
                                                                               usernum,
                                                                               itemnum,
                                                                               batch_size,
                                                                               self.train_queue,
                                                                               self.valid_queue,
                                                                               self.test_queue,
                                                                               np.random.randint(2e9),
                                                                               )))
            self.threads[-1].daemon = True
            self.threads[-1].start()

    def next_train_batch(self):
        return self.train_queue.get()

    def next_valid_batch(self):
        return self.valid_queue.get()

    def next_test_batch(self):
        return self.test_queue.get()
