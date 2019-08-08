from collections import OrderedDict
from utils.path_util import abspath
import torch

# 训练参数
task_name = "cnn_v20"
epochs = 80
batch_size = 64
embed_dim = 200
n_filters = 128
filter_sizes = [3, 4, 5]
n_hidden = 1024
dropout = 0.5
init_lr = 1e-2
lr_decay = 0.96
min_lr = 1e-4
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# csv解析字段以及顺序, x代表文本，y代表标签
data_fields = OrderedDict({
    "content": "x",
    "location_traffic_convenience": "y",
    "location_distance_from_business_district": "y",
    "location_easy_to_find": "y",
    "service_wait_time": "y",
    "service_waiters_attitude": "y",
    "service_parking_convenience": "y",
    "service_serving_speed": "y",
    "price_level": "y",
    "price_cost_effective": "y",
    "price_discount": "y",
    "environment_decoration": "y",
    "environment_noise": "y",
    "environment_space": "y",
    "environment_cleaness": "y",
    "dish_portion": "y",
    "dish_taste": "y",
    "dish_look": "y",
    "dish_recommendation": "y",
    "others_overall_experience": "y",
    "others_willing_to_consume_again": "y",
})
skip_header = True
delimiter = "\x01"

# 训练数据
train_file = abspath("data/sa.train.csv")
valid_file = abspath("data/sa.valid.csv")

# 模型参数数据、
resume = True
user_dict = abspath("library/user.dict")
vector_cache = abspath("library/")
min_freq = 1
use_pre_embed = False
pre_embeddings = abspath("library/embeddings.300w.txt")
extend_vocab = True
pre_vocab_size = 200000

model_file = abspath(f"checkpoints/{task_name}.model.ckpt")
field_file = abspath(f"checkpoints/{task_name}.field.ckpt")
summary_dir = abspath(f"summary/{task_name}/")
log_file = abspath(f"log/{task_name}.log")
