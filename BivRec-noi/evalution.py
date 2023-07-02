from collections import defaultdict
import math
import sys
import time
import faiss
import numpy as np
import torch
import torch.nn as nn
from utils import get_DataLoader, load_model, save_model, load_item_cate, compute_diversity
import torch.nn.functional as F

def InfoNCE_loss(mm_information, id_information, temperature=0.07):
    # Concatenate the multimodal and id representations
    combined = torch.cat((mm_information.unsqueeze(1), id_information.unsqueeze(1)), dim=1)
    batch_size = combined.shape[0]

    # Compute the cosine similarity matrix
    cosine_sim = F.cosine_similarity(combined.unsqueeze(2), combined.unsqueeze(1), dim=-1)

    # Compute the positive pair similarity (diagonal elements)
    pos_sim = torch.diagonal(cosine_sim, dim1=1, dim2=2)  # .reshape(batch_size, 1)

    # Compute the negative pair similarity (off-diagonal elements)
    neg_sim = torch.exp(cosine_sim / temperature).sum(dim=2) - torch.exp(pos_sim / temperature)

    # Compute the loss
    loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + neg_sim)).mean()

    return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, mm, id):
        batch_size = mm.shape[0]

        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=mm.device)

        # normalize mm and id
        mm = F.normalize(mm, dim=-1)
        id = F.normalize(id, dim=-1)

        # compute pairwise cosine similarity between mm and id
        sim_matrix = mm @ id.t()

        # create a mask to exclude samples from the same user
        mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

        # compute the numerator and denominator of the contrastive loss
        numerator = torch.exp(sim_matrix / self.temperature)
        denominator = numerator.sum(dim=-1, keepdim=True) - numerator.masked_select(mask).view(batch_size, -1).sum(dim=-1, keepdim=True)

        # compute the final loss
        loss = -torch.log(numerator / denominator).mean()

        return loss

def to_tensor(var, device):  # list (numpy) --> tensor ()
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()

def to_tensor_tupe(var, device):
    var = torch.stack(var)
    var = var.to(device)
    return var.float()


# cpu:
def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None):
    topN = k  # 评价时选取topN
    if coef is not None:
        coef = float(coef)

    item_embs = model.output_items().cpu().detach().numpy()


    try:
        gpu_index = faiss.IndexFlatL2(hidden_size)
        gpu_index.add(item_embs)  # 给index添加向量数据
    except Exception as e:
        print("error:", e)
        return

# def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None):
#     topN = k  # 评价时选取topN
#     if coef is not None:
#         coef = float(coef)
#
#     item_embs = model.output_items().cpu().detach().numpy()
#
#     res = faiss.StandardGpuResources() # 使用单个GPU
#     flat_config = faiss.GpuIndexFlatConfig()
#     flat_config.device = device.index # 使用GPU
#
#     try:
#         gpu_index = faiss.GpuIndexFlatIP(res, hidden_size, flat_config) # 建立GPU index用于Inner Product近邻搜索
#         # gpu_index = faiss.IndexFlatL2(hidden_size)
#         gpu_index.add(item_embs)  # 给index添加向量数据
#     except Exception as e:
#         print("error:", e)
#         return

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for _, (users, targets, items, imag_his, text_his, mask) in enumerate(test_data):  # 一个batch的数据

        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        user_embs, _, _= model(to_tensor(items, device), None, None, None, to_tensor(mask, device), device, train=False)
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)  # 不重复label物品
                for no, iid in enumerate(I[i]):  # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
                    if iid in true_item_set:  # 如果该推荐物品是label物品
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:  # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(I[i], item_cate_map)  # 两个参数分别为推荐物品列表和物品类别字典
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])  # shape=(batch_size*num_interest, embedding_dim)
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                if coef is None:  # 不考虑物品多样性
                    # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                    item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    item_list.sort(key=lambda x: x[1], reverse=True)  # 降序排序，内积越大，向量越近
                    for j in range(len(item_list)):  # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            item_cor_list.append(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:  # 考虑物品多样性
                    coef = float(coef)
                    # 所有兴趣向量的近邻物品集中起来按距离再次排序
                    origin_item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    origin_item_list.sort(key=lambda x: x[1], reverse=True)
                    item_list = []  # 存放（item_id, distance, item_cate）三元组，要用到物品类别，所以只存放有类别的物品
                    tmp_item_set = set()  # 近邻推荐物品中有类别的物品的集合
                    for (x, y) in origin_item_list:  # x为索引，y为距离
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):  # 选出topN个物品
                        max_index = 0
                        # score = distance - λ * 已选出的物品中与该物品的类别相同的物品的数量（score越大越好）
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):  # 遍历所有候选物品，每个循环找出一个score最大的item
                            # 第一次遍历必然先选出第一个物品
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:  # 当距离得分小于max_score时，后续物品得分一定小于max_score
                                break
                        item_list_set.add(item_list[max_index][0])
                        item_cor_list.append(item_list[max_index][0])
                        # 选出来的物品的类别对应的value加1，这里是为了尽可能选出类别不同的物品
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)  # 候选物品列表中删掉选出来的物品

                # 上述if-else只是为了用不同方式计算得到最后推荐的结果item列表
                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):  # 对于推荐的每一个物品
                    if iid in true_item_set:  # 如果该物品是label物品
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)  # len(iid_list)表示label数量
                if recall > 0:  # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        total += len(targets)  # total增加每个批次的用户数量

    recall = total_recall / total  # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total  # NDCG
    hitrate = total_hitrate * 1.0 / total  # 命中率
    if coef is None:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    diversity = total_diversity * 1.0 / total  # 多样性
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


def train(device, train_file, valid_file, test_file, model, batch_size, seq_len, hidden_size, topN, max_iter, test_iter,  patience, exp_name, lamda1, lamda2, image_file, text_file):


    best_model_path = "best_model/" + exp_name + '/'  # 模型保存路径

    # prepare data
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1, image_path=image_file, text_path=text_file)
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0, image_path=image_file, text_path=text_file)

    model = model.to(device)
    nceloss = ContrastiveLoss()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=lr_decay)

    trials = 0

    print('training begin')
    sys.stdout.flush()

    start_time = time.time()
    try:
        total_loss = 0.0
        iter = 0
        best_metric = 0  # 最佳指标值，在这里是最佳recall值
        # scheduler.step()
        for i, (users, targets, items, image_items, text_items, mask) in enumerate(train_data):
            model.train()
            iter += 1
            optimizer.zero_grad()
            _, scores, id, mm = model(to_tensor(items, device), to_tensor(targets, device), to_tensor_tupe(image_items, device), to_tensor_tupe(text_items, device), to_tensor(mask, device), device)
            loss1 = loss_fn(scores, to_tensor(targets, device))
            loss2 = nceloss(id, mm)
            loss = lamda1 * loss1 + lamda2 * loss2
            loss.backward()
            optimizer.step()
            # if iter%1000==0:
            #     scheduler.step()
            total_loss += loss
            if iter % test_iter == 0:
                model.eval()
                metrics = evaluate(model, valid_data, hidden_size, device, topN)
                log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / test_iter)  # 打印loss
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(exp_name)
                print(log_str)

                # 保存recall最佳的模型
                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        save_model(model, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:  # early stopping
                            print("early stopping!")
                            break

                # 每次test之后loss_sum置零
                total_loss = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                sys.stdout.flush()

            if iter >= max_iter * 1000:  # 超过最大迭代次数，退出训练
                break

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')

    load_model(model, best_model_path)
    model.eval()

    # 训练结束后用valid_data测试一次
    metrics = evaluate(model, valid_data, hidden_size, device, topN)
    print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    # 训练结束后用test_data测试一次
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    metrics = evaluate(model, test_data, hidden_size, device, topN)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def test(device, test_file, cate_file, model,  batch_size, seq_len, hidden_size, topN, exp_name, coef=None, image_file=None, text_file=None):

    best_model_path = "best_model/" + exp_name + '/'  # 模型保存路径A

    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()

    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0, image_path=image_file, text_path=text_file)
    if coef != None:
        item_cate_map = load_item_cate(cate_file)  # 读取物品的类型
        metrics = evaluate(model, test_data, hidden_size, device, topN, coef=coef, item_cate_map=item_cate_map)
    else:
        metrics = evaluate(model, test_data, hidden_size, device, topN, coef=coef, item_cate_map=None)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def output(device, model, exp_name):

    best_model_path = "best_model/" + exp_name + '/'  # 模型保存路径
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()

    item_embs = model.output_items()  # 获取物品嵌入
    np.save('output/' + exp_name + '_emb.npy', item_embs)  # 保存物品嵌入
