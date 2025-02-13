import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置模型仓库 ID 和本地保存路径
repo_id = "Qwen/Qwen2.5-VL-7B-Instruct"  # 模型仓库 ID
local_dir = "../models/Qwenvl"  # 本地保存路径

# 下载完整模型目录
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 避免使用符号链接
    resume_download=True,  # 支持断点续传
    # allow_patterns=["*"],  # 下载所有文件
    # ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "*.ot"],  # 可选：排除某些文件类型
    max_workers=8,  # 并发下载线程数
)
