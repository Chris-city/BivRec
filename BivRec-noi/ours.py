import torch
import torch.nn as nn
import gensim
import faiss
import copy
# from kmeans_pytorch import kmeans
import time
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


# the interest seperation part is reference from GroupVit

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, hard, k: int = 2, tau: float = 1,  dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        # index = y_soft.max(dim, keepdim=True)[1]
        # y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # ret = y_hard - y_soft.detach() + y_soft
        _, top_indices = torch.topk(y_soft, k=k, dim=dim)
        # index = torch.zeros_like(y_soft)
        # index.scatter_(dim, top_indices, 1.0)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, top_indices, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Attention(nn.Module):

    def __init__(self,
                 args,
                 qk_scale=None,
                 qkv_fuse=False):
        super().__init__()
        # self.args = args
        out_dim = args.hidden_size
        self.attn_drop_rate = args.attn_drop
        self.proj_drop_rate = args.proj_drop
        self.num_heads = args.num_heads
        head_dim = args.hidden_size // args.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_fuse = qkv_fuse
        self.qkv_bias = args.qkv_bias

        if qkv_fuse:
            self.qkv = nn.Linear(args.hidden_size, args.hidden_size * 3, bias=self.qkv_bias)
        else:
            self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.qkv_bias)
            self.k_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.qkv_bias)
            self.v_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(args.hidden_size, out_dim)
        self.proj_drop = nn.Dropout(self.proj_drop_rate)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        atten_mask = torch.unsqueeze(mask, dim=1).unsqueeze(dim=2)  # shape=(batch_size, num_heads, 1, lenk)
        paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0
        attn = torch.where(torch.eq(atten_mask, 0), paddings, attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# transformer block
class AttnBlock(nn.Module):

    def __init__(self,
                 args,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(args.hidden_size)
        self.attn = Attention(
            args)
        self.drop_path = nn.Dropout(args.attn_drop) if args.attn_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(args.hidden_size)
        mlp_hidden_dim = int(args.hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=args.hidden_size, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=args.proj_drop)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 args,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        self.hidden_size = args.hidden_size
        if post_norm:
            self.norm_post = norm_layer(self.hidden_size)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(self.hidden_size)
            self.norm_k = norm_layer(self.hidden_size)
            self.norm_post = nn.Identity()
        self.attn = Attention(args)
        self.drop_path = DropPath(args.attn_drop) if args.attn_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=self.hidden_size, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=args.proj_drop)

    def forward(self, query, key, mask):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


# 公式3，4以及5的一部分:qk相乘得到A，经过gumbel hard乘以v，再加上一些trick层
class AssignAttention(nn.Module):

    def __init__(self,
                 args,
                 num_heads=1,
                 qk_scale=None,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = args.hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_bias = args.qkv_bias_assign
        self.attn_drop_rate = args.attn_drop
        self.proj_drop_rate = args.proj_drop
        self.k = args.shelter
        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.qkv_bias)
        self.v_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_drop = nn.Dropout(self.proj_drop_rate)
        self.hard = args.hard
        self.gumbel_tau = args.gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps
        self.gumbel = args.gumbel

        self.assign_mask = args.assign_mask

    def get_attn(self, attn):
        attn_dim = -2
        attn = gumbel_softmax(attn, hard=self.hard, k=self.k, dim=attn_dim, tau=self.gumbel_tau)
        return attn
    def softmax_hard(self, attn):
        attn_dim = -2
        _, top_indices = torch.topk(attn, k=self.k, dim=attn_dim)
        y_hard = torch.zeros_like(attn, memory_format=torch.legacy_contiguous_format).scatter_(attn_dim, top_indices,                                                                                         1.0)
        ret = y_hard - attn.detach() + attn
        return ret


    def forward(self, query, key=None, value=None, mask=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.gumbel == 'open':
            attn = self.get_attn(raw_attn)
        elif self.gumbel == 'close' and self.hard:  #softmax and hard
            attn = self.softmax_hard(raw_attn)
        else:
            attn = raw_attn
        # hard gumbel

        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)

        if self.assign_mask is not None:
            atten_mask = torch.unsqueeze(mask, dim=1).unsqueeze(dim=2)
            paddings = torch.ones_like(atten_mask, dtype=torch.float) * (-2 ** 32 + 1)  # softmax之后无限接近于0
            attn = torch.where(torch.eq(atten_mask, 0), paddings, attn)
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        # 断言表达式，if true 程序才继续运行
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict


class GroupingBlock(nn.Module):

    def __init__(self,
                 args,
                 *,
                 norm_layer=nn.LayerNorm,
                 sum_assign=False):
        super(GroupingBlock, self).__init__()
        self.dim = args.hidden_size
        self.hard = args.hard
        self.sum_assign = sum_assign
        # self.num_output_group = num_output_group
        self.if_project = args.if_project
        self.if_cross = args.if_cross
        # norm on group_tokens and x
        self.norm_tokens = norm_layer(args.hidden_size)

        self.mlp_inter = Mlp(args.interest_num, 4 * args.interest_num, args.interest_num, drop=args.special_drop)
        self.norm_post_tokens = norm_layer(args.hidden_size)
        self.pre_assign_attn = CrossAttnBlock(args)

        self.assign = AssignAttention(
            args)
        self.norm_new_x = norm_layer(args.hidden_size)
        self.reduction = nn.Identity()

    def project_group_token(self, group_tokens):
        if self.if_project:
            projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
            projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
            return projected_group_tokens

        else:
            return group_tokens

    def pre_information(self, projected_group_tokens, x, mask):
        if self.if_cross:
            infor_group_tokens = self.pre_assign_attn(projected_group_tokens, x, mask)
            return infor_group_tokens
        else:
            return projected_group_tokens

    def forward(self, group_tokens, x, mask, return_attn=False):
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_tokens(x)
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_information(projected_group_tokens, x, mask)
        new_x, attn_dict = self.assign(query=projected_group_tokens, key=x, mask=mask, return_attn=return_attn)
        new_x += projected_group_tokens
        new_x = self.norm_new_x(new_x)

        return new_x, attn_dict


class Trans_depth_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # layer = AttnBlock(args)
        # self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_trans_layers)])
        self.layers = nn.ModuleList([AttnBlock(args) for _ in range(args.num_trans_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MMhard_small(nn.Module):
    def __init__(self, args):
        super(MMhard_small, self).__init__()
        self.item_num = args.item_num
        self.seq_len = args.seq_len
        self.pos_drop = nn.Dropout(args.pos_drop)
        self.item_num = args.item_num
        self.hidden_size = args.hidden_size
        self.num_group_token = args.interest_num
        self.hard_readout = args.hard_readout
        self.embeddings = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.pos_embed = args.add_pos
        self.device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if args.MLPtoVector:
            self.head = Mlp(args.hidden_size, 4 * args.hidden_size, args.hidden_size, drop=args.special_drop)
        else:
            self.head = nn.Identity()
        # self.crossatten = CrossAttnBlock(args)
        self.assignment = GroupingBlock(args)
        self.transformerencoder = Trans_depth_layer(args)
        self.target_norm = nn.LayerNorm(args.hidden_size)
        self.tokens_norm = nn.LayerNorm(args.hidden_size)
        self.pass_type = args.pass_type
        # self.throughtranslayer = args.throughtranslayer
        self.group_token = nn.Parameter(torch.zeros(1, args.interest_num, args.hidden_size))
        if args.trunc:
            trunc_normal_(self.group_token, std=.02)

        if args.init_self:
            self.apply(self.init_weights)
        else:
            self.reset_parameters()

    def split_x(self, x):
        return x[:, :self.num_group_token], x[:, self.num_group_token:]

    def concat_x(self, x, group_token):
        return torch.cat([x, group_token], dim=1)

    def add_pos_sep(self, x):
        if not self.pos_embed:
            return x
        else:
            return self.pos_drop(
                self.norm(x + nn.Parameter(torch.zeros(1, x.size(1), self.hidden_size)).to(self.device)))

    def unit_vector(self, x):
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def trough_trans(self, x, mask):
        x = self.transformerencoder(x, mask, output_all_encoded_layers=True)
        output = x[-1]
        return output

    # input mask(batch,sequence_len)----> output mask(batch, sequence_len + num_interest)
    # so that, when we do add transformer layer to concat sequence, the mask can ignore the padding 0 in history items.
    def pad_mask(self, mask):
        pad = torch.ones((mask.size(0), self.num_group_token), dtype=torch.long, device=mask.device)
        padded_mask = torch.cat((pad, mask), dim=1)
        return padded_mask

    def information_passing(self, tokens, x, mask):
        if self.pass_type == 'togather':
            cat_x = self.concat_x(tokens, x)
            mask_add = self.pad_mask(mask)
            cat_x = self.trough_trans(cat_x, mask_add)
            group_token, x = self.split_x(cat_x)
        elif self.pass_type == 'self':
            x = self.trough_trans(x, mask)
            group_token = tokens
        elif self.pass_type == 'dual':
            x = self.trough_trans(x, mask)
            cat_x = self.concat_x(tokens, x)
            mask_add = self.pad_mask(mask)
            cat_x = self.trough_trans(cat_x, mask_add)
            group_token, x = self.split_x(cat_x)
        elif self.pass_type == 'no':
            group_token = tokens
            x = x
        return group_token, x

    def forward(self, input_ids, target_item, mask, device, train=True):

        item_eb = self.embeddings(input_ids)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            target_eb = self.target_norm(self.embeddings(target_item))
        item_emb_pos = self.add_pos_sep(item_eb)
        group_token_0 = self.group_token.expand(item_emb_pos.size(0), -1, -1)
        group_token, x = self.information_passing(group_token_0, item_emb_pos, mask)
        fin_group, attn_dict = self.assignment(group_token, x, mask)
        fin_group = self.tokens_norm(fin_group)
        # vector = self.unit_vector(fin_group)

        if not train:
            return fin_group, None

        readout = self.read_out(fin_group, target_eb)
        scores = self.calculate_score(readout)

        return fin_group, scores

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    ################################Basic_model#########################
    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)

    def read_out(self, user_eb, label_eb):

        # 这个模型训练过程中label是可见的，此处的item_eb就是label物品的嵌入
        atten = torch.matmul(user_eb,  # shape=(batch_size, interest_num, hidden_size)
                             torch.reshape(label_eb, (-1, self.hidden_size, 1))  # shape=(batch_size, hidden_size, 1)
                             )  # shape=(batch_size, interest_num, 1)
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.num_group_token)), 1),
                          dim=-1)  # shape=(batch_size, interest_num)

        if self.hard_readout:  # 选取interest_num个兴趣胶囊中的一个，MIND和ComiRec都是用的这种方式
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + torch.arange(label_eb.shape[0],
                                                            device=user_eb.device) * self.num_group_token).long()]
        else:  # 综合interest_num个兴趣胶囊，论文及代码实现中没有使用这种方法
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.num_group_token)),
                                   # shape=(batch_size, 1, interest_num)
                                   user_eb  # shape=(batch_size, interest_num, hidden_size)
                                   )  # shape=(batch_size, 1, hidden_size)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size))  # shape=(batch_size, hidden_size)
        # readout是vu堆叠成的矩阵（一个batch的vu）（vu可以说就是最终的用户嵌入）
        return readout

    def calculate_score(self, user_eb):
        all_items = self.embeddings.weight
        scores = torch.matmul(user_eb, all_items.transpose(1, 0))  # [b, n]
        return scores

    def output_items(self):
        return self.embeddings.weight

