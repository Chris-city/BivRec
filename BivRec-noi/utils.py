import os
import random
import shutil
import numpy as np
import torch

from torch.utils.data import DataLoader

# parser = get_parser()
# args = parser.parse_args()


class DataIterator(torch.utils.data.IterableDataset):
    def __init__(self, source,
                 batch_size=2048,
                 seq_len=20,
                 train_flag=1,
                 image_feature_file=None,
                 text_feature_file=None
                 ):
        self.read(source)  # 读取数据，获取用户列表和对应的按时间戳排序的物品序列，每个用户对应一个物品list
        self.users = list(self.users)  # 用户列表
        self.batch_size = batch_size  # 用于训练
        self.eval_batch_size = batch_size  # 用于验证、测试
        self.train_flag = train_flag  # train_flag=1表示训练
        self.seq_len = seq_len  # 历史物品序列的最大长度
        self.index = 0  # 验证和测试时选择用户的位置的标记
        print("total user:", len(self.users))

        # Load image and text features
        self.image_features = np.load(image_feature_file)
        self.text_features = np.load(text_feature_file)

    def __iter__(self):
        return self

    def read(self, source):
        self.graph = {}  # key:user_id，value:一个list，放着该user_id所有(item_id,time_stamp)元组，排序后value只保留item_id
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
            for i, line in enumerate(f):
                try:
                    conts = line.strip().split(',')
                    user_id = int(conts[0])
                    item_id = int(conts[1])
                    time_stamp = int(conts[3])
                    self.users.add(user_id)
                    self.items.add(item_id)
                    if user_id not in self.graph:
                        self.graph[user_id] = []
                    self.graph[user_id].append((item_id, time_stamp))
                except ValueError as e:
                    print(f"Error in line {i + 1}: {line.strip()}")
                    print('error:',line)
                    print(f"Exception: {e}")
        for user_id, value in self.graph.items():  # 每个user的物品序列按时间戳排序
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [x[0] for x in value]  # 排序后只保留了item_id
        self.users = list(self.users)  # 用户列表
        self.items = list(self.items)  # 物品列表

    def __next__(self):
        if self.train_flag == 1:  # 训练
            user_id_list = np.random.choice(self.users, self.batch_size)  # 随机抽取batch_size个user
        else:  # 验证、测试，按顺序选取eval_batch_size个user，直到遍历完所有user
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = [] # 选取的item_id
        hist_item_list = [] # 选取的item_id的历史序列
        hist_mask_list = [] # 选取的item_id的历史序列的mask
        image_his_list = [] # 选取的item_id的历史序列的image特征
        image_target = [] # 选取的item_id的image特征
        text_his_list = []
        text_target = []
        for user_id in user_id_list:
            item_list = self.graph[user_id]  # 排序后的user的item序列
            if self.train_flag == 1:  # 训练，选取训练时的label
                k = random.choice(range(4, len(item_list)))  # 从[4,len(item_list))中随机选择一个index
                item_id_list.append(item_list[k])  # 该index对应的item加入item_id_list
            else:  # 验证、测试，选取该user后20%的item用于验证、测试
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            if k >= self.seq_len:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
            image_his_list.append(self.image_features[hist_item_list[-1]])
            text_his_list.append(self.text_features[hist_item_list[-1]])
        return user_id_list, item_id_list, hist_item_list, image_his_list, text_his_list, hist_mask_list


def get_DataLoader(source, batch_size, seq_len, train_flag, image_path, text_path):
    dataIterator = DataIterator(source, batch_size, seq_len, train_flag, image_feature_file=image_path, text_feature_file=text_path)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 生成实验名称
# def get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=True):
def get_exp_name(args, save=True):
    # extr_name = input('Please input the experiment name: ')
    extr_name = '_'.join(['project' + str(args.if_project), 'trans' + str(args.throughtranslayer),  'pos' + str(args.add_pos), 'assignmask' + str(args.assign_mask)])
    para_name = '_'.join([args.dataset, 'b' + str(args.batch_size), 'lr' + str(args.learning_rate), 'd' + str(args.hidden_size),
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


# 读取物品类别信息，返回一个dict，key:item_id，value:cate_id
def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


# 计算物品多样性，item_list中的所有item两两计算
def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n - 1) * n / 2)
    return diversity
