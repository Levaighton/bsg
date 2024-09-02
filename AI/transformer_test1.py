import torch
import torch.nn as nn
import torch.optim as optim
import math

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        return output, attention

# Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        bs = Q.size(0)

        Q = self.q_linear(Q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(K).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(V).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores, attention = ScaledDotProductAttention(self.d_k)(Q, K, V, mask)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        output = self.out(concat)

        return output, attention

# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, input_vocab_size, max_len):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, output_vocab_size, max_len):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        output = self.fc_out(x)
        return output

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Transformer Model (Encoder-Decoder)
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, input_vocab_size, max_len, output_vocab_size):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, input_vocab_size, max_len)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff, output_vocab_size, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

# Dictionary to simulate vocabulary mappings (in real use, these would be learned during training)
english_to_index = {"hello": 0, "world": 1}
chinese_to_index = {"你好": 0, "世界": 1}
index_to_chinese = {0: "你好", 1: "世界"}

# Simulated input and output tokens (in a real scenario, you'd have a tokenized input)
def translate_word(word, model):
    src_token = torch.tensor([english_to_index[word]]).unsqueeze(0)  # Add batch dimension
    tgt_token = torch.tensor([chinese_to_index["你好"]]).unsqueeze(0)  # Assume we start with "你好"

    output = model(src_token, tgt_token)
    output_index = torch.argmax(output, dim=-1).item()

    return index_to_chinese[output_index]

# Training loop
def train_model(model, data, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        for src_sentence, tgt_sentence in data:
            src_tokens = torch.tensor([english_to_index[word] for word in src_sentence]).unsqueeze(0)
            tgt_tokens = torch.tensor([chinese_to_index[word] for word in tgt_sentence]).unsqueeze(0)

            optimizer.zero_grad()
            output = model(src_tokens, tgt_tokens[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt_tokens[:, 1:].view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/len(data):.4f}')

# Example usage
if __name__ == "__main__":
    # Example parameters
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    input_vocab_size = len(english_to_index)
    output_vocab_size = len(chinese_to_index)
    max_len = 100

    # Instantiate the model
    model = Transformer(d_model, num_heads, num_layers, d_ff, input_vocab_size, max_len, output_vocab_size)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example data (you would use a real dataset here)
    data = [(["hello"], ["你好"]), (["world"], ["世界"]), (["Fuck"],["草泥马的傻逼"])]

    # Train the model
    train_model(model, data, criterion, optimizer, num_epochs=10)

    # User input loop for translation
    while True:
        english_word = input("Enter an English word to translate (or type 'exit' to quit): ").strip().lower()
        if english_word == 'exit':
            break
        if english_word in english_to_index:
            chinese_translation = translate_word(english_word, model)
            print(f"The translation of '{english_word}' is '{chinese_translation}'")
        else:
            print(f"The word '{english_word}' is not in the vocabulary.")
