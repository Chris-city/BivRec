import os
import sys
from ours import MMhard_small
import argparse

pid = os.getpid()
print('pid:%d' % (pid))

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
CUDA_LAUNCH_BLOCKING=1

import torch
from utils import setup_seed, get_exp_name
from evalution import train, test, output


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        value = getattr(args, arg)
        if value is None:
            value_str = "None"
        else:
            value_str = f"{value}"
        print(f"{arg:<30} : {value_str:>35}")

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--log_path', default='./results/test.log', type=str, help='log file path to save result')
    parser.add_argument('--p', type=str, default='train', help='train | test')  # train or test or output
    parser.add_argument('--dataset', type=str, default='ml-1m', help='book | taobao')  # 数据集
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--test_iter', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='MMhard_small', help='DNN | GRU4Rec | MIND | ..')  # 模型类型
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')  # 学习率
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=30,
                        help='(k), the number of steps after which the learning rate decay')
    parser.add_argument('--max_iter', type=int, default=500, help='(k)')  # 最大迭代次数，单位是k（1000）
    parser.add_argument('--patience', type=int, default=50)  # patience，用于early stopping
    parser.add_argument('--topN', type=int, default=20)  # default=50
    parser.add_argument('--gpu', type=str, default='0')  # None -> cpu
    parser.add_argument('--coef', default=None)  # 多样性，用于test
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    ##########
    parser.add_argument("--coef_center", type=float, default=0.3, help="center loss coef")
    parser.add_argument("--start_center", default=20, help="start to do center_loss")
    parser.add_argument("--center_loss", default=True, help="if use center CL loss")
    parser.add_argument("--shelter", default=1, type=int, help="the shelter k in hard gradient")
    parser.add_argument("--gumbel", default='open', type=str, help="if use gumbel softmax instead softmax | close")
    parser.add_argument("--gumbel_tau", default=1., type=float, help="if hard, it should 1.0, else small like 0.5/0.1")
    parser.add_argument("--trunc", default=True, type=bool, help="wether to initial group_tokens")
    parser.add_argument("--hard", default=True, type=bool, help="if hard01")
    parser.add_argument("--pass_type", default='no', type=str, help="togather or self to through transformer")
    parser.add_argument("--hard_readout", default=True, help="chose the closest interest")
    parser.add_argument("--assign_mask", default=True, help="if mask in assign_attention")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--num_trans_layers", default=1, type=int,
                        help="additional trans layer for information passing")
    parser.add_argument("--throughtranslayer", default=False, type=bool, help="if through trans of group and x")
    parser.add_argument("--qkv_bias", default=False, type=bool, help="for translayer when the dataset is small and model is large, it should be False")
    parser.add_argument("--qkv_bias_assign", default=True, type=bool, help="in Group Block, we want the complexity of model")
    # parser.add_argument("--mlp_bias", default=True, help="all projection bias")  fix it in the mlp function
    parser.add_argument("--attn_drop", default=0.5, type=float)
    parser.add_argument("--proj_drop", default=0.5, type=float, help="feed foward")
    parser.add_argument("--special_drop", default=0.5, type=float, help="for head and mlp_inter")
    parser.add_argument("--pos_drop", default=0.5, type=float, help="position drop ratio")
    parser.add_argument("--item_num", default=3417, type=int, help="see main.py: 103993sports |23033cloth |3417ml")
    parser.add_argument("--seq_len", default=200, type=int)
    # parser.add_argument("--dim", default=64, type=int, help="embedding size")
    parser.add_argument("--num_heads", default=2, type=int, help="multi-head for MMhard")
    parser.add_argument("--double_sep", default=None, help="whether to mining interest once or twice")
    parser.add_argument('--interest_num', type=int, default=4)  # 兴趣的数量
    parser.add_argument('--if_cross', default=False, type=bool, help="whether do cross attention")
    parser.add_argument("--if_project", default=False, type =bool, help="whether project by mlp and only zero init group token if we have a projection")
    parser.add_argument("--add_pos", default=True, type=bool, help="whether to add position embedding")
    parser.add_argument("--MLPtoVector", default=False, type=bool,
                        help="add mlp in the final vector to map to different space between ID and MM")
    parser.add_argument("--init_self", default=True, type=bool, help="whether to init by nn.kaiming or truncted normal")
    # parser.add_argument("--pos_embed_type", default='simple', type=str, help="simple or fourier")
    # parser.add_argument("--dpr", default=None)
    # parser.add_argument("--addsubsequent", default=None, help="whether to add subsequent mask")
    # parser.add_argument("--mask_twice", default=True, type=bool, help="whether add subsequent mask ")

    args = parser.parse_args()



    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu:
        device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        print("use cuda:" + args.gpu if torch.cuda.is_available() else "use cpu, cuda:" + args.gpu + " not available")
    else:
        device = torch.device("cpu")
        print("use cpu")

    SEED = args.random_seed
    setup_seed(SEED)

    if args.dataset == 'book':
        path = './data/book_data/'
        # item_count = 367982 + 1 / 703121 + 1
        # user = 1855618
        # interaction = 27158711
        # batch_size = args.batch_size
        # avg_len = 14.6
        # test_iter = args.test_iter
    elif args.dataset == 'ml-1m':
        path = './data/ml_data/'
        center_update = 10
        # item_count = 3417
        # users = 6040
        # interactions = 1000000
        # avg_len = 165
        # batch_size = args.batch_size
        # seq_len = 200
        # drop = 0.2
    elif args.dataset == 'beauty':
        path = './data/beauty_data/'
        # item_count = 8862 + 1
        # users = 1274
        # interactions = 7113
        # avg_len = 5.5
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'sports':
        path = './data/sports_data/'
        # item_count = 103992 + 1
        # users = 331919
        # interactions = 2835746
        # avg_len = 8.5
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'luxy':
        path = './data/luxy_data/'
        # item_count = 1438 + 1
        # users = 3680
        # interactions = 33362
        # avg_len = 9
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'fashion':
        path = './data/fashion_data/'
        # item_count = 30120
        # users = 1912
        # interactions = 12463
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'familyTV':
        path = './data/familyTV_data/'
        # item_count = 867632 + 1
        # batch_size = args.batch_size
        # seq_len = 30
        # test_iter = args.test_iter
    elif args.dataset == 'kindle':
        path = './data/kindle_data/'
        # item_count = 260154 + 1
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'taobao':
        path = './data/taobao_data/'
        # item_count = 1708531
        # batch_size = 256
        # maxlen = 50
        # test_iter = 500
    elif args.dataset == 'toy':
        path = './data/toy_data/'
        # item_count = 78205 + 1
        # users = 207812
        # interactions = 1825836
        # avg_len = 9
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'sport':
        path = './data/sport_data/'
        # item_count = 18356 + 1
        # users = 35597
        # interactions = 296337
        # avg_len = 8
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'elec':
        path = './data/elec_data/'
        # item_count = 63000 + 1
        # users = 192402
        # interactions = 1689188
        # avg_len = 8
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'cloth':
        path = './data/cloth_data/'
        # item_count = 23032 + 1
        # users = 39386
        # interactions = 278677
        # avg_len = 7
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter
    elif args.dataset == 'baby':
        path = './data/baby_data/'
        center_update = 10
        # item_count = 7049 + 1
        # users = 19444
        # interactions = 160792
        # avg_len = 8
        # batch_size = args.batch_size
        # seq_len = 20
        # test_iter = args.test_iter

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'

    print("hidden_size:", args.hidden_size)
    print("interest_num:", args.interest_num)
    show_args_info(args)

    exp_name = get_exp_name(args,save=False)
    model = MMhard_small(args)

    # center_update is the iteration to do E-step in EM algorithm to update the center of each cluster
    if args.p == 'train':
        train(device=device, train_file=train_file, valid_file=valid_file, test_file=test_file, model=model,batch_size=args.batch_size,
              seq_len=args.seq_len, hidden_size=args.hidden_size, topN=args.topN, max_iter=args.max_iter,
              test_iter=args.test_iter,patience=args.patience, exp_name=exp_name,
              start_center=args.start_center, lambda_=args.coef_center, center_update=center_update)

    elif args.p == 'test':
        test(device=device, test_file=test_file, cate_file=cate_file, model=model, batch_size=args.batch_size, seq_len=args.seq_len,
             hidden_size=args.hidden_size, topN=args.topN, exp_name=exp_name, coef=args.coef)
    elif args.p == 'output':
        output(device=device, model=model, exp_name=exp_name)
    else:
        print('do nothing...')



main()




