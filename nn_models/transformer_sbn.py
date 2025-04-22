import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, bptt):
        super().__init__()
        self.positional_embedding = nn.Embedding(bptt, embedding_size)
    def forward(self, x):
        N, S = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S))
        x = self.positional_embedding(position)
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout, bptt):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size, bptt)
        self.embedding = nn.Embedding(num_tokens + 1, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        src = self.embedding(src) + self.positional_embedding(src)
        src = self.dropout(self.norm(src))
        return src

class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    def forward(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_o = nn.Linear(embedding_size, embedding_size)
        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)
    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)
    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature).permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)
    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.linear_o(q)
        return q, attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.activation = nn.GELU()
        self.init_param()
    def init_param(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        return
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# class Decoder(nn.Module):
#     def __init__(self, num_tokens, embedding_size):
#         super().__init__()
#         self.linear1 = nn.Linear(embedding_size, embedding_size)
#         self.activation = nn.GELU()
#         self.norm1 = nn.LayerNorm(embedding_size)
#         self.linear2 = nn.Linear(embedding_size, embedding_size)
#         self.norm2 = nn.LayerNorm(embedding_size)
#         self.linear3 = nn.Linear(embedding_size, num_tokens)

#     def forward(self, src):
#         out = self.norm1(self.activation(self.linear1(src)))
#         out = self.norm2(self.activation(self.linear2(out)))
#         out = self.linear3(out)
#         return out

class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size):
        super().__init__()

        self.linear1 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear1(src)
        return out

class Multi_Transformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size, dropout, bptt):
        super().__init__()
        self.num_tokens = num_tokens
        self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout, bptt)
        self.encoder_layer1 = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.encoder_layer2 = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.encoder_layer3 = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.encoder_layer4 = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.decoder4 = Decoder(num_tokens, embedding_size)
        

    def forward(self, input):
        src = input.clone().type(torch.IntTensor).to(input.device)
        
        mask = torch.rand(input.shape) < 0.15
        mask_change = mask & (torch.rand(input.shape) < 0.9)
        mask_random = mask_change & (torch.rand(input.shape) < 1/9)

        src[mask_change] = self.num_tokens
        src.masked_scatter_(mask_random.to(input.device), torch.randint(1, self.num_tokens, mask_random.shape, dtype=torch.int).to(input.device))
        src = src.detach()

        src = self.transformer_embedding(src)
        src = self.encoder_layer1(src)
        src = self.encoder_layer2(src)
        src = self.encoder_layer3(src)
        src = self.encoder_layer4(src)
        out = self.decoder4(src)

        # return out
        return out, mask

def multi_transformer(width=1.0):

    # below from heteroFL paper
    
    num_tokens = 28782
    embedding_size = int(512 * width)
    num_heads = 8
    hidden_size = int(512 * width)
    dropout = 0.2
    bptt = 64  # not sure
    return Multi_Transformer(num_tokens, embedding_size, num_heads, hidden_size, dropout, bptt)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    
    model = multi_transformer(width=4/4)
    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (64,), as_strings=True,
                                                print_per_layer_stat=False, verbose=True, units='GMac')

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    
    
