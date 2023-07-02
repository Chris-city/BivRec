import os
import random
import shutil
import numpy as np
import torch

from torch.utils.data import DataLoader


class DataIterator(torch.utils.data.IterableDataset):

    def __init__(self, source,
                 batch_size=2048,
                 seq_len=20,
                 train_flag=1
                 ):
        self.read(source)  
        self.users = list(self.users)  

        self.batch_size = batch_size  
        self.eval_batch_size = batch_size  
        self.train_flag = train_flag  
        self.seq_len = seq_len 
        self.index = 0  
        print("total user:", len(self.users))

    def __iter__(self):
        return self

    # def next(self):
    #     return self.__next__()

    # for amazon:

    # def read(self, source):
    #     self.graph = {}  
    #     self.users = set()
    #     self.items = set()
    #     with open(source, 'r') as f:
    #         for line in f:
    #             conts = line.strip().split(',')
    #             user_id = int(conts[0])
    #             item_id = int(conts[1])
    #             time_stamp = int(conts[2])

    #             self.users.add(user_id)
    #             self.items.add(item_id)
    #             if user_id not in self.graph:
    #                 self.graph[user_id] = []
    #             self.graph[user_id].append((item_id, time_stamp))
    #     for user_id, value in self.graph.items():  
    #         value.sort(key=lambda x: x[1])
    #         self.graph[user_id] = [x[0] for x in value] 
    #     self.users = list(self.users) 
    #     self.items = list(self.items)  

    # for ml-1m:
    # def read(self, source):
    #     self.graph = {}  
    #     self.users = set()
    #     self.items = set()
    #     with open(source, 'r') as f:
    #         for line in f:
    #             conts = line.strip().split(' ')
    #             user_id = int(conts[0])
    #             item_id = int(conts[1])
    #
    #             self.users.add(user_id)
    #             self.items.add(item_id)
    #             if user_id not in self.graph:
    #                 self.graph[user_id] = []
    #             self.graph[user_id].append(item_id)
    #     self.users = list(self.users)  
    #     self.items = list(self.items)  
    #
    # for multimodal:

    def read(self, source):
        self.graph = {}  
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[3])

                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items():  
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [x[0] for x in value]  
        self.users = list(self.users)  
        self.items = list(self.items)  

    def __next__(self):
        if self.train_flag == 1:  
            user_id_list = random.sample(self.users, self.batch_size)  
        else: 
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        for user_id in user_id_list:
            item_list = self.graph[user_id]  
            
            if self.train_flag == 1:  
                
                k = random.choice(range(4, len(item_list)))
                # k = len(item_list)-1
                item_id_list.append(item_list[k])  
            else:  
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            
            if k >= self.seq_len:  
                hist_item_list.append(item_list[k - self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))

       
        return user_id_list, item_id_list, hist_item_list, hist_mask_list


def get_DataLoader(source, batch_size, seq_len, train_flag=1):
    dataIterator = DataIterator(source, batch_size, seq_len, train_flag)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# def get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=True):
def get_exp_name(args, save=True):
    # extr_name = input('Please input the experiment name: ')
    extr_name = '_'.join(['project' + str(args.if_project), 'trans' + str(args.pass_type), 'pos' + str(args.add_pos),
                          'assignmask' + str(args.assign_mask), 'cross' + str(args.if_cross)])
    para_name = '_'.join(
        [args.dataset, 'b' + str(args.batch_size), 'lr' + str(args.learning_rate), 'd' + str(args.hidden_size),
         'len' + str(args.seq_len), 'in' + str(args.interest_num), 'top' + str(args.topN)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('best_model/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('best_model/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


def save_model(model, Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')


def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'))
    print('model loaded from %s' % path)


def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()



def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate



def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n - 1) * n / 2)
    return diversity
