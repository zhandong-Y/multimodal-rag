# import os
# import shutil

# # 检查这些路径是否有坏文件
# search_paths = [
#     os.path.expanduser("~/.nltk_data/tokenizers/punkt.zip"),
#     os.path.expanduser("~/.nltk_data/tokenizers/punkt"),
#     "/home/amax/nltk_data/tokenizers/punkt.zip",
#     "/home/amax/anaconda3/envs/multimodal-rag1-env/nltk_data/tokenizers/punkt.zip",
#     "/home/amax/anaconda3/envs/multimodal-rag1-env/nltk_data/tokenizers/punkt",
# ]

# # 设置是否实际删除
# dry_run = True  # True 表示只打印，不实际删除

# for path in search_paths:
#     if os.path.exists(path):
#         if dry_run:
#             print(f"[DRY RUN] Would delete: {path}")
#         else:
#             if os.path.isfile(path):
#                 os.remove(path)
#                 print(f"Deleted file: {path}")
#             else:
#                 shutil.rmtree(path)
#                 print(f"Deleted directory: {path}")
#     else:
#         print(f"Not found: {path}")
import nltk
import shutil
import os

# 设置 NLTK 数据路径
nltk_data_path = "/home/amax/nltk_data"
nltk.data.path = [nltk_data_path]


# 重新下载所需的数据
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
