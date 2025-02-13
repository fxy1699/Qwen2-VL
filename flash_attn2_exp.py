import torch
from flash_attn import flash_attn_func

# 检查 CUDA 是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")

# 设置设备
device = torch.device("cuda")

# 创建随机输入张量
batch_size = 2
seq_len = 128
num_heads = 8
head_dim = 64

query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

# 调用 Flash-Attn2
output = flash_attn_func(query, key, value, causal=False)

print("Output shape:", output.shape)