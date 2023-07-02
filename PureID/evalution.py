from collections import defaultdict
import math
import sys
import time
import faiss
import numpy as np
import torch
import torch.nn as nn
from utils import get_DataLoader, load_model, save_model, to_tensor, load_item_cate, compute_diversity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(model, test_data, device, name):
    for _, (users, targets, items, mask) in enumerate(test_data):  


        user_embs, _ = model(to_tensor(items, device), None, to_tensor(mask, device), device, train=False)
        user_embs = user_embs.cpu().detach().numpy() # shape=(batch_size, num_interest, embedding_dim)
        np.save(name, user_embs)
        tsne_batch = user_embs.shape[0]
        selected_users_idx = np.random.choice(tsne_batch, size=20, replace=False)
        selected_users_embs = user_embs[selected_users_idx]
        selected_users_embs_2d = selected_users_embs.reshape((-1, selected_users_embs.shape[-1]))
        tsne = TSNE(n_components=2)
        selected_users_embs_2d = tsne.fit_transform(selected_users_embs_2d)
        np.save(f"tsne/{name}_2d.npy", selected_users_embs_2d)
        color_map = plt.get_cmap('tab20') 
        color_idx = 0
        for i, user_idx in enumerate(selected_users_idx):
            user_embss = user_embs[user_idx]
            num_interests = user_embss.shape[0]
            user_color = color_map(i / len(selected_users_idx))
            for j in range(num_interests):
                plt.scatter(selected_users_embs_2d[color_idx, 0], selected_users_embs_2d[color_idx, 1], color=user_color, s=50)
                color_idx += 1
        user_labels = [f"User {i}" for i in range(len(selected_users_idx))]
        plt.legend(user_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        # add axis labels and title
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization of User Interests')

        plt.subplots_adjust(right=0.7)
        plt.show()
        plt.savefig(name, bbox_inches='tight')

# cpu:
def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None, iter = None):
    topN = k 
    if coef is not None:
        coef = float(coef)

    item_embs = model.output_items().cpu().detach().numpy()


    try:
        gpu_index = faiss.IndexFlatL2(hidden_size)
        gpu_index.add(item_embs)  
    except Exception as e:
        print("error:", e)
        return

# def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None):
#     topN = k  
#     if coef is not None:
#         coef = float(coef)
#
#     item_embs = model.output_items().cpu().detach().numpy() 
#
#     res = faiss.StandardGpuResources() 
#     flat_config = faiss.GpuIndexFlatConfig()
#     flat_config.device = device.index 
#
#     try:
#         gpu_index = faiss.GpuIndexFlatIP(res, hidden_size, flat_config) 
#         # gpu_index = faiss.IndexFlatL2(hidden_size)
#         gpu_index.add(item_embs)  
#     except Exception as e:
#         print("error:", e)
#         return

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for _, (users, targets, items, mask) in enumerate(test_data):  

        user_embs, _ = model(to_tensor(items, device), None, to_tensor(mask, device), device, train=False)
        user_embs = user_embs.cpu().detach().numpy() # shape=(batch_size, num_interest, embedding_dim)
        if len(user_embs.shape) == 2: 
            D, I = gpu_index.search(user_embs, topN) 
            for i, iid_list in enumerate(targets):  
       
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)  
                for no, iid in enumerate(I[i]):  
                    if iid in true_item_set:  
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0: 
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(I[i], item_cate_map)  
        else:  
  
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])  
            D, I = gpu_index.search(user_embs, topN) 
            for i, iid_list in enumerate(targets):  
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                if coef is None: 
                    
                    item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    item_list.sort(key=lambda x: x[1], reverse=True)  
                    for j in range(len(item_list)): 
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            item_cor_list.append(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:  
                    coef = float(coef)
                  
                    origin_item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    origin_item_list.sort(key=lambda x: x[1], reverse=True)
                    item_list = []  
                    tmp_item_set = set()  
                    for (x, y) in origin_item_list: 
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):  
                        max_index = 0
                        
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):  
                            
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:  
                                break
                        item_list_set.add(item_list[max_index][0])
                        item_cor_list.append(item_list[max_index][0])
                        
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)  

            
                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list): 
                    if iid in true_item_set: 
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)  
                if recall > 0:  
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        total += len(targets)  

    recall = total_recall / total  
    ndcg = total_ndcg / total  # NDCG
    hitrate = total_hitrate * 1.0 / total 
    if coef is None:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    diversity = total_diversity * 1.0 / total  
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


def train(device, train_file, valid_file, test_file, model, batch_size, seq_len, hidden_size, topN, max_iter, test_iter,  patience, exp_name):


    best_model_path = "best_model/" + exp_name + '/'  

    # prepare data
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1)
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0)

    model = model.to(device)

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
        best_metric = 0  
        # scheduler.step()
        for i, (users, targets, items, mask) in enumerate(train_data):
            model.train()
            iter += 1
            optimizer.zero_grad()
            _, scores = model(to_tensor(items, device), to_tensor(targets, device), to_tensor(mask, device), device)
            loss = loss_fn(scores, to_tensor(targets, device))
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

           
                total_loss = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                sys.stdout.flush()

            if iter >= max_iter * 1000:  
                break

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')

    load_model(model, best_model_path)
    model.eval()


    tsne_plot(model, valid_data, device, name='valid')
    metrics = evaluate(model, valid_data, hidden_size, device, topN)
    print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

 
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    tsne_plot(model, test_data, device, name='test')
    metrics = evaluate(model, test_data, hidden_size, device, topN)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def test(device, test_file, cate_file, model,  batch_size, seq_len, hidden_size, topN, exp_name, coef=None):

    best_model_path = "best_model/" + exp_name + '/'  

    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()

    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    if coef != None:
        item_cate_map = load_item_cate(cate_file)  
        metrics = evaluate(model, test_data, hidden_size, device, topN, coef=coef, item_cate_map=item_cate_map)
    else:
        metrics = evaluate(model, test_data, hidden_size, device, topN, coef=coef, item_cate_map=None)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def output(device, model, exp_name):

    best_model_path = "best_model/" + exp_name + '/'  
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()

    item_embs = model.output_items()  
    np.save('output/' + exp_name + '_emb.npy', item_embs)  
