import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from ours import Mlp, GroupingBlock, Trans_depth_layer


class VanillaAttention(nn.Module):

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class MMclus_noi(nn.Module):
    def __init__(self, args):
        super(MMclus_noi, self).__init__()
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
        self.norm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.avgpool_id = nn.AdaptiveAvgPool1d(1)
        self.avgpool_mm = nn.AdaptiveAvgPool1d(1)
        self.mm_type = args.mm_type
        self.modality = args.modality
        self.MLPtoVector = args.MLPtoVector
        if args.MLPtoVector:
            self.head_id = Mlp(args.hidden_size, 4 * args.hidden_size, args.hidden_size, drop=args.special_drop)
            self.head_mm = Mlp(args.hidden_size, 4 * args.hidden_size, args.hidden_size, drop=args.special_drop)
        else:
            self.head_id = nn.Identity()
            self.head_mm = nn.Identity()
        self.assignment = GroupingBlock(args)
        self.transformerencoder = Trans_depth_layer(args)
        self.target_norm = nn.LayerNorm(args.hidden_size)
        self.tokens_norm = nn.LayerNorm(args.hidden_size)
        self.pass_type = args.pass_type
        self.group_token = nn.Parameter(torch.zeros(1, args.interest_num, args.hidden_size))
        self.direct = args.direct
        self.text_layer = nn.Linear(384, args.mm_dim)
        self.image_layer = nn.Linear(4096, args.mm_dim)
        self.both_dim = nn.Linear(128, args.mm_dim)

        if not args.direct:
            self.mm_dim = args.mm_dim
            self.norm_mm = nn.LayerNorm(args.mm_dim, eps=1e-12)
            self.mm_tokens = nn.Parameter(torch.zeros(1, args.interest_num, args.mm_dim))
            self.assignment_mm = GroupingBlock(args)

        if args.trunc:
            trunc_normal_(self.group_token, std=.02)

        if args.init_self:
            self.apply(self.init_weights)
        else:
            self.reset_parameters()

    def split_x(self, x):
        return x[:, :self.num_group_token], x[:, self.num_group_token:]

    def concat_x(self, x, y):
        return torch.cat([x, y], dim=1)

    def add_pos_sep(self, x):
        if not self.pos_embed:
            return x
        else:
            return self.pos_drop(
                self.norm(x + nn.Parameter(torch.zeros(1, x.size(1), self.hidden_size)).to(self.device)))

    def add_pos_mm(self, x):
        return self.pos_drop(
            self.norm_mm(x + nn.Parameter(torch.zeros(1, x.size(1), x.size(2))).to(self.device)))

    def unit_vector_id(self, x):
        x = self.avgpool_id(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head_id(x)
        return x

    def unit_vector_mm(self, x):
        x = self.avgpool_mm(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head_mm(x)
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

    def non_invasive(self, image_ids, txt_ids, mask):
        """
        both image and text are not in this code
        """
        if self.direct:
            if self.modality == 'image':
                image_ids = self.image_layer(image_ids)
                image_ids = self.tokens_norm(image_ids)  # B N 64
                mm = self.unit_vector_mm(image_ids)
            elif self.modality == 'text':
                txt_ids = self.text_layer(txt_ids)
                txt_ids = self.tokens_norm(txt_ids)  # B N 64
                mm = self.unit_vector_mm(txt_ids)
            elif self.modality == 'both':
                if self.mm_type == 'concat':
                    image_ids = self.image_layer(image_ids)
                    txt_ids = self.text_layer(txt_ids)
                    mm = torch.cat([image_ids, txt_ids], dim=-1)
                    mm = self.both_dim(mm)
                    mm = self.tokens_norm(mm) # B N 64
                    mm = self.unit_vector_mm(mm)
                elif self.mm_type == 'sum':
                    image_ids = self.image_layer(image_ids)
                    txt_ids = self.text_layer(txt_ids)
                    mm = image_ids + txt_ids
                    mm = self.tokens_norm(mm) # B N 64
                    mm = self.unit_vector_mm(mm)
            return mm
        else:  # through interest block
            if self.modality == 'image':
                image_ids = self.image_layer(image_ids)
                mm_pos = self.add_pos_mm(image_ids)
                mm_token = self.mm_tokens.expand(mm_pos.size(0), -1, -1)
                mm_token, _ = self.assignment_mm(mm_token, mm_pos, mask)
                mm = self.tokens_norm(mm_token)
                mm = self.unit_vector_mm(mm)
            elif self.modality == 'text':
                txt_ids = self.text_layer(txt_ids)
                mm_pos = self.add_pos_mm(txt_ids)
                mm_token = self.mm_tokens.expand(mm_pos.size(0), -1, -1)
                mm_token, _ = self.assignment_mm(mm_token, mm_pos, mask)
                mm = self.tokens_norm(mm_token)
                mm = self.unit_vector_mm(mm)
            elif self.modality == 'both':
                image_ids = self.image_layer(image_ids)
                txt_ids = self.text_layer(txt_ids.float())
                if self.mm_type == 'concat':
                    mm = torch.cat([image_ids, txt_ids], dim=-1)
                    mm_eb = self.both_dim(mm)
                    mm_pos = self.add_pos_mm(mm_eb)
                    mm_token = self.mm_tokens.expand(mm_pos.size(0), -1, -1)
                    mm_token, _ = self.assignment_mm(mm_token, mm_pos, mask)
                    mm = self.tokens_norm(mm_token)
                    mm = self.unit_vector_mm(mm)
                elif self.mm_type == 'sum':
                    mm_eb = image_ids + txt_ids
                    mm_pos = self.add_pos_mm(mm_eb)
                    mm_token = self.mm_tokens.expand(mm_pos.size(0), -1, -1)
                    mm_token, _ = self.assignment_mm(mm_token, mm_pos, mask)
                    mm = self.tokens_norm(mm_token)
                    mm = self.unit_vector_mm(mm)
            return mm

    def forward(self, input_ids, target_item, image_ids, txt_ids, mask, device, train=True):
        item_eb = self.embeddings(input_ids)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))
        if train:
            target_eb = self.target_norm(self.embeddings(target_item))
            mm_vector = self.non_invasive(image_ids, txt_ids, mask)  # 经过兴趣聚类器并转换成一个向量

        item_emb_pos = self.add_pos_sep(item_eb)
        group_token_0 = self.group_token.expand(item_emb_pos.size(0), -1, -1)
        group_token, x = self.information_passing(group_token_0, item_emb_pos, mask)
        fin_group, attn_dict = self.assignment(group_token, x, mask)
        fin_group = self.tokens_norm(fin_group)
        id_vector = self.unit_vector_id(fin_group)

        if not train:
            return fin_group, None, None

        readout = self.read_out(fin_group, target_eb)
        scores = self.calculate_score(readout)
        return fin_group, scores, id_vector, mm_vector

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

