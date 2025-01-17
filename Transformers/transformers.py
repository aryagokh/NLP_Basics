import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert(self.head_dim * heads == embed_size), "Embedded size needs to be divisible by heads!"

        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch Size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # Number of words (seq_len)
        
        # (N, seq_len, embed_size) @ (N, embed_size, embed_size) -> (N, seq_len, embed_size)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Reshaping for multi head attention
        # (N, seq_len, embed_size) -> (N, seq_len, heads, head_dim). Example (1, 10, 12288) -> (1, 10, 96, 128)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Ensure value_len matches key_len before attention calculation
        assert value_len == key_len, "Value length must match key length!"

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # Query * Key Transpose
        if mask is not None:
            energy = energy.masked_fill(mask==0, value=float('-1e-20'))

        attention = torch.softmax(energy / (self.head_dim**0.5), dim=3)
        out = torch.einsum('nhqk,nhkd->nqhd', [attention, values]).reshape(N, query_len, self.embed_size)

        return self.fc_out(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        attention = self.dropout(attention)
        attention = self.norm1(attention + query)

        forward = self.feed_forward(attention)
        forward = self.dropout(forward)
        out = self.norm2(forward + attention)

        return out
    


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, 
                 forward_expansion, dropout, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embed_size)
        self.position_encoding = nn.Embedding(num_embeddings=max_length, embedding_dim=embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ 
            TransformerBlock(embed_size=embed_size, heads=heads, 
                             dropout=dropout, forward_expansion=forward_expansion) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.word_embedding(x) + self.position_encoding(positions)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size=embed_size, heads=heads, 
                                                  dropout=dropout, 
                                                  forward_expansion=forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        
        return out
    


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, num_layers,
                 heads, forward_expansion, dropout, device, max_length):
        super().__init__()
        self.device = device
        self.word_embeddings = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
             )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.word_embeddings(x) + self.position_embedding(positions)
        out = self.dropout(out)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        out = self.fc_out(x)
        return out



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx,
                 embed_size=256, num_layers=8, forward_expansion=4, heads=8,
                 dropout=0, device="cpu", max_length=100):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, 
            embed_size, num_layers, 
            heads, 
            device, 
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            tgt_vocab_size, 
            embed_size, 
            num_layers, 
            heads, 
            forward_expansion, 
            dropout, 
            device, 
            max_length
        )
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
        return tgt_mask.to(self.device)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out
    


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor([[4, 6, 8, 1, 0, 4, 7, 5, 9], [5, 2, 6, 2, 5, 2, 7, 7, 5]]).to(device)
    tgt = torch.tensor([[3, 2, 5, 1, 6, 7, 3, 2, 1], [5, 5, 4, 5, 7, 8, 9, 2, 1]])
    src_pad_idx = 0
    tgt_pad_idx = 0
    src_vocab_size=10
    tgt_vocab_size=10
    
    model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx).to(device)
    out = model(x, tgt[:, :-1])  # Forward pass
    print(out.shape)