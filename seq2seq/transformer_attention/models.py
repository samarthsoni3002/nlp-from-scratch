import torch 
from torch import nn 
from torch.nn import functional as F


def positional_encoding(seq_len, d_model):

  pe = torch.zeros(seq_len, d_model)

  positions = torch.arange(0, seq_len).unsqueeze(1)

  div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

  pe[:, 0::2] = torch.sin(positions * div_term)
  pe[:, 1::2] = torch.cos(positions * div_term)

  return pe

def scaled_dot_product_attention(query, key, value, mask=None):

  d_k = key.size(-1)

  dot_prod = query.matmul(key.transpose(-2,-1)) 

  scale_factor = torch.sqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))
  scores = dot_prod / scale_factor

  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

  attention_weights = F.softmax(scores,dim=-1) 

  output = attention_weights.matmul(value) 
  
  return output, attention_weights


class MultiHeadAttention(nn.Module):

  def __init__(self, num_head, d_model):
    super().__init__()

    assert d_model % num_head == 0

    self.d_model = d_model
    self.num_head = num_head
    self.d_k = d_model // num_head

    self.wq = nn.Linear(d_model, d_model)
    self.wk = nn.Linear(d_model, d_model)
    self.wv = nn.Linear(d_model, d_model)

    self.wo = nn.Linear(d_model, d_model)

  def forward(self, query, key, value, mask=None):

    batch_size = query.size(0)

    query = self.wq(query) # [batch_size, seq_len, d_model]
    key = self.wk(key) # [batch_size, seq_len, d_model]
    value = self.wv(value) # [batch_size, seq_len, d_model]

    query = query.view(batch_size, -1, self.num_head, self.d_k).transpose(1,2) # [batch_size, num_heads, seq_len, d_k]
    key = key.view(batch_size, -1, self.num_head, self.d_k).transpose(1,2) # [batch_size, num_heads, seq_len, d_k]
    value = value.view(batch_size, -1, self.num_head, self.d_k).transpose(1,2) # [batch_size, num_heads, seq_len, d_k]

    output, attention_weights = scaled_dot_product_attention(query, key, value, mask) # output = [batch_size, num_head, seq_len, d_v (same as d_k)], attention_weights = [batch_size, num_heads,seq_len, seq_len]

    output = output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model) # [batch_size, seq_len, d_model]

    output = self.wo(output)  # [batch_size, seq_len, d_model]

    return output, attention_weights


class Encoder(nn.Module):

  def __init__(self, d_model, hidden_dim, vocab_size, max_len, positional_encoding,MultiHeadAttention, num_heads,pad_idx=0):
    super().__init__()


    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.max_len = max_len
    self.pad_idx = pad_idx
    self.num_heads = num_heads
    self.d_model = d_model

    self.pos_encoding = positional_encoding
    self.multiheadattention = MultiHeadAttention(num_heads, d_model)


    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d_model)
    )


  def forward(self, input_ids,mask):

    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    embed = self.embedding(input_ids) # [batch_size, seq_len, embed_dim]

    positions = self.pos_encoding(seq_len, self.d_model).to(input_ids.device) # [batch_size, embed_dim]

    final_emb = embed + positions.unsqueeze(0)

    mha_output, attention_weights = self.multiheadattention(final_emb,final_emb,final_emb,mask)

    total_output = mha_output + final_emb

    norm1 = self.norm1(total_output)

    fc = self.feed_forward(norm1)

    total_fc_output = fc + norm1

    output = self.norm2(total_fc_output)

    return output, attention_weights


class Decoder(nn.Module):

  def __init__(self, d_model, hidden_dim, vocab_size, max_len, positional_encoding,MultiHeadAttention, num_heads,pad_idx=0):
    super().__init__()


    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.max_len = max_len
    self.pad_idx = pad_idx
    self.num_heads = num_heads
    self.d_model = d_model

    self.pos_encoding = positional_encoding
    self.self_attention = MultiHeadAttention(num_heads, d_model)
    self.cross_attention = MultiHeadAttention(num_heads, d_model)


    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)

    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d_model)
    )

  def create_mask(self, output_ids):

    batch_size, seq_len = output_ids.shape

    pad_mask = (output_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), device=output_ids.device)
    ).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    tgt_mask = pad_mask & causal_mask

    return tgt_mask


  def forward(self, output_ids, encoder_outputs,src_mask):

    batch_size = output_ids.size(0)
    seq_len = output_ids.size(1)

    embed = self.embedding(output_ids) # [batch_size, seq_len, embed_dim]

    positions = self.pos_encoding(seq_len, self.d_model).to(output_ids.device) # [batch_size, embed_dim]

    final_emb = embed + positions.unsqueeze(0)

    mask = self.create_mask(output_ids)

    mha_output, self_attention_weights = self.self_attention(final_emb,final_emb,final_emb,mask)

    total_output = mha_output + final_emb

    norm1 = self.norm1(total_output)

    mha_output, cross_attention_weights = self.cross_attention(norm1,encoder_outputs,encoder_outputs,src_mask)

    total_output = mha_output + norm1

    norm2 = self.norm2(total_output)

    fc = self.feed_forward(norm2)

    total_fc_output = fc + norm2

    output = self.norm3(total_fc_output)

    return output, self_attention_weights, cross_attention_weights


class Transformer(nn.Module):

    def __init__(self, encoder, decoder, d_model, vocab_size, pad_idx=0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def create_src_mask(self, input_ids):
        return (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids, output_ids):

        src_mask = self.create_src_mask(input_ids)

        encoder_outputs, enc_attn = self.encoder(input_ids, src_mask)

        decoder_outputs, self_attn, cross_attn = self.decoder(
            output_ids,
            encoder_outputs,
            src_mask
        )

        logits = self.fc_out(decoder_outputs)

        return logits