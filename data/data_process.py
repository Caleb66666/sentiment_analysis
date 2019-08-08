import pandas as pd
from utils.path_util import abspath


train_file = abspath("data/sa_train.csv")
valid_file = abspath("data/sa_valid.csv")
test_file = abspath("data/sa_test.csv")


train_df = pd.read_csv(train_file, header=0, sep="\t")
valid_df = pd.read_csv(valid_file, header=0, sep="\t")
test_df = pd.read_csv(test_file, header=0, sep="\t")


train_df.to_csv(abspath("data/sa.train.csv"), index=False, sep="\x01")
valid_df.to_csv(abspath("data/sa.valid.csv"), index=False, sep="\x01")
test_df.to_csv(abspath("data/sa.test.csv"), index=False, sep="\x01")
