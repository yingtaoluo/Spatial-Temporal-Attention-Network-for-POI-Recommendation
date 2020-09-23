from STAN.load import *
import time
from torch import optim
import torch.utils.data as data
from STAN.models import *

# os.chdir("C:\\Users\\罗颖涛\\PycharmProjects\\POI")
os.chdir("C:\\Users\\Administrator\\PycharmProjects\\POI")


def calculate_acc(log_prob, label):
    # log_prob (N, L), label (N), batch_size [*M]
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(log_prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            # topk_predict (k)
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)


def generate_balanced_weight(label, num_neg, l_dim):
    init_weight = np.zeros(shape=(l_dim,))
    # num_neg samples from [1, l_max]
    random_ig = random.sample(range(1, l_dim + 1), num_neg)
    while len([lab for lab in label if lab in random_ig]) != 0:
        random_ig = random.sample(range(1, l_dim + 1), num_neg)

    for lab in label:  # positive
        init_weight[lab] = 1
    for ig in random_ig:  # selected negative
        init_weight[ig] = 1

    return torch.FloatTensor(init_weight)


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index]
        mats1 = self.mat1[index]
        vector = self.vec[index]
        label = self.label[index]
        length = self.length[index]
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


class Trainer:
    def __init__(self, model, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.iteration = record['iterations'][-1] if load else 0
        self.num_neg = 10
        self.interval = 100
        self.batch_size = 1
        self.learning_rate = 3e-3
        self.epoch = 1000
        self.threshold = record['acc_valid'][-1] if load else 0  # 0 if not update

        # (NUM, M+2, 3), (NUM, M+2, M+2, 2), (L, L), (NUM, M+2), tensor (NUM, 3), (NUM, [*M])
        self.traj, self.mat1, self.mat2, self.vec, self.label, self.len = trajs, mat1, mat2, vec, labels, lens
        self.train_set = DataSet(self.traj[:, :-2], self.mat1[:, :-2, :-2],
                                 self.vec[:, :-2], self.label[:, 0], self.len)
        self.train_loader = data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)

        self.valid_set = DataSet(self.traj[:, 1:-1], self.mat1[:, 1:-1, 1:-1],
                                 self.vec[:, 1:-1], self.label[:, 1], self.len)
        self.valid_loader = data.DataLoader(dataset=self.valid_set, batch_size=self.batch_size, shuffle=True)
        self.valid_iter = iter(self.valid_loader)

        self.test_set = DataSet(self.traj[:, 2:], self.mat1[:, 2:, 2:],
                                self.vec[:, 2:], self.label[:, 2], self.len)
        self.test_loader = data.DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=True)
        self.test_iter = iter(self.test_loader)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.epoch):
            # begin training
            for step, item in enumerate(self.train_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M), (N), [N]
                train_input, train_m1, train_v, train_label, train_traj_len = item

                # use encoder input to replace encoder output as decoder input while training
                log_prob = self.model(train_input, train_m1, self.mat2, train_v, train_traj_len)  # (N, L)
                train_weight = generate_balanced_weight(train_label, self.num_neg, l_max)
                loss_train = F.cross_entropy(log_prob, train_label, weight=train_weight, ignore_index=0)

                if self.iteration % self.interval is 0:
                    end = time.time()
                    print('iteration: {}, time: {}'.format(self.iteration, end - start))

                    # training loss & acc
                    print('train_loss: {:.4}'.format(loss_train))
                    acc_train = calculate_acc(log_prob, train_label) / self.batch_size
                    print('train_acc', acc_train)

                    # valid loss & acc
                    loss_valid = 0
                    valid_size = 0
                    acc_valid = [0, 0, 0, 0]
                    while True:
                        try:
                            valid_input, valid_m1, valid_v, valid_label, valid_traj_len = self.valid_iter.next()
                            valid_size += valid_input.shape[0]
                        except StopIteration:
                            self.valid_iter = iter(self.valid_loader)
                            break

                        # loss calculation
                        log_prob_valid = self.model(valid_input, valid_m1, self.mat2, valid_v, valid_traj_len)
                        valid_weight = generate_balanced_weight(valid_label, self.num_neg, l_max)
                        loss_valid += F.cross_entropy(log_prob_valid, valid_label, weight=valid_weight,
                                                      ignore_index=0, reduction='sum')
                        acc_valid += calculate_acc(log_prob_valid, valid_label)

                    loss_valid = to_npy(loss_valid) / valid_size
                    acc_valid = np.array(acc_valid) / valid_size
                    print('valid_loss: {:.4}'.format(loss_valid))
                    print('valid_acc:', acc_valid)

                    # test loss & acc
                    loss_test = 0
                    test_size = 0
                    acc_test = [0, 0, 0, 0]
                    while True:
                        try:
                            test_input, test_m1, test_v, test_label, test_traj_len = self.test_iter.next()
                            test_size += test_input.shape[0]
                        except StopIteration:
                            self.test_iter = iter(self.test_loader)
                            break

                        # loss calculation
                        log_prob_test = self.model(test_input, test_m1, self.mat2, test_v, test_traj_len)
                        test_weight = generate_balanced_weight(test_label, self.num_neg, l_max)
                        loss_test += F.cross_entropy(log_prob_test, test_label, weight=test_weight,
                                                     ignore_index=0, reduction='sum')
                        acc_test += calculate_acc(log_prob_test, test_label)

                    loss_test = to_npy(loss_test) / test_size
                    acc_test = np.array(acc_test) / test_size
                    print('test_loss: {:.4}'.format(loss_test))
                    print('test_acc:', acc_test)

                    self.records['acc_train'].append(acc_train)
                    self.records['acc_valid'].append(acc_valid)
                    self.records['acc_test'].append(acc_test)
                    self.records['iterations'].append(self.iteration)

                    if self.threshold > np.mean(acc_test):
                        self.threshold = np.mean(acc_test)
                        # save the model
                        torch.save({'state_dict': self.model.state_dict(),
                                    'records': self.records,
                                    'time': time.time() - start},
                                   'best_model_stan.pth')

                # update parameters
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                loss_train.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.iteration += 1
                scheduler.step()


if __name__ == '__main__':
    # load data
    dname = 'NYC'
    file = open('./data/' + dname + '_data.pkl', 'rb')
    file_data = joblib.load(file)
    # tensor(NUM, M+2, 3), list(NUM, M+2, M+2, 2), list(L, L),
    # list(NUM, M+2), tensor(NUM, 3), list(NUM), (1), (1)
    [trajs, mat1, mat2, vec, labels, lens, u_max, l_max] = file_data
    mat1, mat2, vec, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2), \
                            torch.FloatTensor(vec), torch.LongTensor(lens)
    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()

    stan = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=80, ex=ex, dropout=0)

    load = False
    train = True

    if load:
        checkpoint = torch.load('best_model_stan.pth')
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'recall': [], 'iterations': [],
                   'acc_train': [], 'acc_valid': [], 'acc_test': []}
        start = time.time()

    trainer = Trainer(stan, records)
    trainer.train()

