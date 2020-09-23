from ours.load import *
import time
from torch import optim
import torch.utils.data as data
from ours.models import *


class DataSet(data.Dataset):
    def __init__(self, input_tensor, input_len, label_tensor):
        self.input_tensor = input_tensor  # (N, ?, 3)
        self.input_len = input_len  # (N)
        self.label_tensor = label_tensor  # (N)

    def __getitem__(self, index):
        input_data = self.input_tensor[index]
        input_len = self.input_len[index]
        label_data = self.label_tensor[index]
        return input_data, input_len, label_data

    def __len__(self):
        return len(self.input_tensor)


# sort batch according to length and get length
def collate_fn(batch):
    batch.sort(key=lambda x: x[1], reverse=True)  # x is element in list
    inputs, lens, labels = [], [], []
    for i, item in enumerate(batch):
        inputs.append(batch[i][0].numpy().tolist())
        lens.append(batch[i][1])
        labels.append(batch[i][2].numpy().tolist())
    inputs = torch.LongTensor(inputs).to(device)
    lens = torch.LongTensor(lens).to(device)
    labels = torch.LongTensor(labels).to(device)
    return inputs, lens, labels  # nll_loss requires target 0 <= target <= C - 1


class Trainer:
    def __init__(self, model, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.iteration = record['iterations'][-1] if load else 0
        self.iter_num = record['iterations']
        self.losses = {'loss_train': record['loss_train'],
                       'loss_test': record['loss_test']}
        self.interval = 100
        self.batch_size = 32
        self.learning_rate = 3e-3
        self.epoch = 1000
        self.threshold = record['loss_valid'][-1] if load else np.inf  # 0 if not update

        # load dataset
        self.train_pad, self.test_pad = train_pad, test_pad  # (N, max, 3)
        self.train_len, self.test_len = train_length, test_length  # (N)

        self.train_label = torch.LongTensor(train_labels)  # (N)
        self.test_label = torch.LongTensor(test_labels)

        self.dataset = DataSet(self.train_pad, self.train_len, self.train_label)
        self.train_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                            shuffle=True, collate_fn=collate_fn)

        self.test_set = DataSet(self.test_pad, self.test_len, self.test_label)
        self.test_loader = data.DataLoader(dataset=self.test_set, batch_size=self.batch_size,
                                            shuffle=True, collate_fn=collate_fn)
        self.test_iter = iter(self.test_loader)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=3e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.epoch):
            # begin training
            for step, item in enumerate(self.train_loader):
                # get batch data
                train_input, train_traj_len, train_label = item  # (N, max_seq_len, 3), [N], (N)

                # use encoder input to replace encoder output as decoder input while training
                log_prob = self.model(train_input, train_traj_len)  # (N, l_dim)
                loss_train = F.cross_entropy(log_prob, train_label)

                if self.iteration % self.interval is 0:
                    end = time.time()
                    print('iteration: {}, time: {}'.format(self.iteration, end - start))
                    print('train_loss: {:.4}'.format(loss_train))

                    acc_train = [0, 0, 0, 0, 0]
                    for i, k in enumerate([1, 5, 10, 20, 50]):
                        _, topk_predict_batch = torch.topk(log_prob, k=k)
                        for topk_predict in to_npy(topk_predict_batch):
                            if to_npy(train_label)[0] in topk_predict:
                                acc_train[i] += 1 / self.batch_size
                    print('acc_train', acc_train)

                # record the training process w.r.t interval
                if self.iteration % self.interval == 0:
                    loss_test, test_size, iter_num = 0, 0, 0
                    acc_test = [0, 0, 0, 0, 0]
                    while True:
                        try:
                            test_input, test_traj_len, test_label = self.test_iter.next()
                            test_N = test_input.shape[0]
                            test_size += test_N
                            iter_num += 1
                        except StopIteration:
                            self.test_iter = iter(self.test_loader)
                            break

                        # loss calculation
                        log_prob_test = self.model(test_input, test_traj_len)  # (N, l_dim)
                        loss_test_batch = F.cross_entropy(log_prob_test, test_label)
                        loss_test += loss_test_batch

                        # topk accuracy calculation
                        for i, k in enumerate([1, 5, 10, 20, 50]):
                            _, topk_predict_batch_test = torch.topk(log_prob_test, k=k)
                            for topk_predict_test in to_npy(topk_predict_batch_test):
                                if to_npy(test_label)[0] in topk_predict_test:
                                    acc_test[i] += 1 / self.batch_size

                    loss_test /= iter_num
                    print('iter_num', iter_num)
                    acc_test = np.array(acc_test) / iter_num

                    end = time.time()
                    print('iteration: {}, time: {}'.format(self.iteration, end-start))
                    print('test_loss: {:.4}'.format(loss_test))
                    print('test_acc:', acc_test)

                    self.iter_num.append(self.iteration)
                    self.losses['loss_train'].append(to_npy(loss_train))
                    self.records['loss_train'] = self.losses['loss_train']
                    self.losses['loss_test'].append(to_npy(loss_test))
                    self.records['loss_test'] = self.losses['loss_test']
                    self.records['iterations'] = self.iter_num

                    if self.threshold > np.mean(acc_test):
                        self.threshold = np.mean(acc_test)
                        # save the model
                        torch.save({'state_dict': self.model.state_dict(),
                                    'records': self.records,
                                    'time': time.time() - start},
                                   'best_model_ours.pth')

                # update parameters
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                loss_train.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.iteration += 1
                scheduler.step()


if __name__ == '__main__':
    # load data
    dn = 'TKY'
    adj, dis = load_adj(dn)  # (7873, [?])
    poi = np.load('./data/' + dn + '_POI.npy')
    [train_trajs, test_trajs], [train_labels, test_labels] = load_traj(dn)  # (N, ?, 3), (N)

    train_trajs.sort(key=lambda x: len(x[0]), reverse=True)
    test_trajs.sort(key=lambda x: len(x[0]), reverse=True)
    train_length = [len(sq) for sq in train_trajs]
    test_length = [len(sq) for sq in test_trajs]

    # (N, max, 3)
    train_pad = pad_sequence(train_trajs, batch_first=True, padding_value=0)
    test_pad = pad_sequence(test_trajs, batch_first=True, padding_value=0)
    max_len = train_pad.shape[1]

    # padding zeros to make train & test equal length
    if train_pad.shape[1] > test_pad.shape[1]:
        padded = torch.zeros((test_pad.shape[0], train_pad.shape[1]-test_pad.shape[1], 3), dtype=torch.long)
        test_pad = torch.cat((test_pad, padded), dim=1)
    elif train_pad.shape[1] < test_pad.shape[1]:
        padded = torch.zeros((train_pad.shape[0], test_pad.shape[1]-train_pad.shape[1], 3), dtype=torch.long)
        train_pad = torch.cat((train_pad, padded), dim=1)
        max_len = test_pad.shape[1]

    u_dim = train_trajs.copy()[-1][0][0].data.numpy()  # true dim
    print(u_dim)

    # group = group_region(dn, train_trajs, u_dim)  # (user, *num, ?)
    group_file = open('./data/' + dn + '_group.txt', 'rb')
    group = pickle.load(group_file)

    model = Model(num_node=len(adj)+1, num_feat=10, t_dim=24+1, l_dim=len(adj)+1, u_dim=u_dim+1, max_len=max_len,
                  embed_dim=100, num_layer=1, hidden_dim=64, adj=adj, dis=dis, poi=poi, g=group, dropout=0)

    load = False
    train = True

    if load:
        checkpoint = torch.load('best_model_ours.pth')
        model.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'err_train': [], 'err_test': [],
                   'loss_train': [], 'loss_test': [],
                   'iterations': []}
        start = time.time()

    trainer = Trainer(model, records)
    trainer.train()

