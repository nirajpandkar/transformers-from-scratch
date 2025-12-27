import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax

# I'm learning to code the attention mechanism myself. I am writing the following function to do so. Generate me a sample query, key and value matrix for me to visualize the calculations i'm doing and nothing more. Don't give me any code. I'll be using the sample as 
# a reference to code.
# Sample query, key, value tensors for visualization (shapes shown)
# batch_size = 1, num_heads = 1, seq_len = 3, d_k = d_v = 4
#
# Query (1,1,3,4):
# [[[[1.0, 0.0, 1.0, 0.0],
#    [0.0, 1.0, 0.0, 1.0],
#    [1.0, 1.0, 0.0, 0.0]]]]
#
# Key (1,1,3,4):
# [[[[1.0, 0.0, 0.0, 1.0],
#    [0.0, 1.0, 1.0, 0.0],
#    [1.0, 1.0, 1.0, 0.0]]]]
#
# Value (1,1,3,4):
# [[[[1.0, 2.0, 3.0, 4.0],
#    [2.0, 3.0, 4.0, 5.0],
#    [3.0, 4.0, 5.0, 6.0]]]]


class MultiHeadAttention(nn.Module):
    #Let me write the initializer just for this class, so you get an idea of how it needs to be done
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" #think why?

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Note: use integer division //

        # Create the learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model) #think why we are doing from d_model -> d_model
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask=None):
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional mask to prevent attention to certain positions
        """

        query = torch.tensor(query)
        key = torch.tensor(key)
        value = torch.tensor(value)
        # how do i get the d_k from the above query matrix?
        d_k = query.size(-1)
        # calculate the attention score using the formula given. Be vary of the dimension of Q and K. And what you need to transpose to achieve the desired results.
        qk = torch.matmul(query, key.transpose(2,3))/math.sqrt(d_k)
        print(f"Multiplying Query and Key matrices: \n{qk})")
        # hint 1: batch_size and num_heads should not change
        # hint 2: nXm @ mXn -> nXn, but you cannot do nXm @ nXm, the right dimension of the left matrix should match the left dimension of the right matrix. The easy way I visualize it is as, who face each other must be same

        # add inf is a mask is given, This is used for the decoder layer. You can use help for this if you want to. I did!!
        #YOUR CODE HERE
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float('-inf'))


        # get the attention weights by taking a softmax on the scores, again be wary of the dimensions. You do not want to take softmax of batch_size or num_heads. Only of the values. How can you do that?
        #YOUR CODE HERE
        softmax_qk = softmax(qk, dim=-1)
        print(f"After taking softmax: \n{softmax_qk}")

        scaled_attention = softmax_qk.matmul(value)
        print(f"Scaled Attention: \n{scaled_attention}")
        
        return scaled_attention

    def forward(self, query, key, value, mask=None):
        #get batch_size and sequence length
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 1. Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply attention
        output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5. Final projection
        return self.W_o(output)
    
class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network

    Args:
        d_model: input/output dimension
        d_ff: hidden dimension
        dropout: dropout rate (default=0.1)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        #create a sequential ff model as mentioned in section 3.3
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.model(x)