from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import Vectors
from config import user_dict
from utils.text_util import pretreatment
from utils.ml_util import init_unk
from functools import partial
import jieba_fast as jieba
import torch

jieba.setLogLevel(20)
jieba.load_userdict(user_dict)


class BatchWrapper(object):
    def __init__(self, batch_iter, x_var, y_vars):
        self.batch_iter = batch_iter
        self.x_var = x_var
        self.y_vars = y_vars

    def __iter__(self):
        for batch in self.batch_iter:
            x, lengths = getattr(batch, self.x_var)

            y_tensors = [getattr(batch, y_var).unsqueeze(1) for y_var in self.y_vars]
            y = torch.cat(tuple(y_tensors), dim=1)

            yield x, y, lengths

    def __len__(self):
        return len(self.batch_iter)


class DataLoader(object):
    def __init__(self, data_fields, train_file, valid_file, batch_size, device, skip_header, delimiter, pre_embeddings,
                 vector_cache, min_freq=2, extend_vocab=True, pre_vocab_size=200000):
        self.x_field = Field(sequential=True, tokenize=self.word_tokenize, batch_first=True, include_lengths=True)
        self.y_field = LabelField(batch_first=True)
        self.train_fields, self.x_var, self.y_vars = self.parse_fields(data_fields, self.x_field, self.y_field)

        self.train_ds = TabularDataset(train_file, fields=self.train_fields, skip_header=skip_header, format="csv",
                                       csv_reader_params={"delimiter": delimiter})
        self.valid_ds = TabularDataset(valid_file, fields=self.train_fields, skip_header=skip_header, format="csv",
                                       csv_reader_params={"delimiter": delimiter})

        vectors = Vectors(pre_embeddings, vector_cache)
        self.x_field.build_vocab(self.train_ds, min_freq=min_freq)
        if extend_vocab:
            unk_init = partial(init_unk, vocab_size=pre_vocab_size)
            self.extend_vocab_with_vectors(self.x_field.vocab, vectors, pre_vocab_size)
        else:
            unk_init = partial(init_unk, vocab_size=len(self.x_field.vocab))
        vectors.unk_init = unk_init
        self.x_field.vocab.load_vectors(vectors)
        self.y_field.build_vocab(self.train_ds)

        self.train_iter, self.valid_iter = BucketIterator.splits(
            (self.train_ds, self.valid_ds),
            batch_size=batch_size,
            device=device,
            sort=False,
            sort_key=lambda sample: len(getattr(sample, self.x_var)),
            sort_within_batch=False,
            shuffle=True,
            repeat=False,
        )

        self.vocab = self.x_field.vocab
        self.vocab_size = len(self.x_field.vocab)
        self.num_labels = len(self.y_vars)
        self.num_classes = len(self.y_field.vocab)
        self.classes = list(self.y_field.vocab.stoi.values())
        self.unk_token = self.x_field.unk_token
        self.pad_token = self.x_field.pad_token
        self.unk_idx = self.x_field.vocab.stoi[self.unk_token]
        self.pad_idx = self.x_field.vocab.stoi[self.pad_token]
        self.train_wrapper = BatchWrapper(self.train_iter, self.x_var, self.y_vars)
        self.valid_wrapper = BatchWrapper(self.valid_iter, self.x_var, self.y_vars)

    @staticmethod
    def word_tokenize(text):
        text = pretreatment(text)
        return jieba.lcut(text)

    @staticmethod
    def char_tokenize(text):
        text = pretreatment(text)
        return list(text)

    @staticmethod
    def parse_fields(data_fields, x_field, y_field):
        train_fields, x_var, y_vars = [], None, []
        for field_name, var_type in data_fields.items():
            if var_type == "x":
                x_var = field_name
                train_fields.append((field_name, x_field))
            elif var_type == "y":
                y_vars.append(field_name)
                train_fields.append((field_name, y_field))
            else:
                train_fields.append((field_name, None))
        return train_fields, x_var, y_vars

    @staticmethod
    def extend_vocab_with_vectors(vocab, vectors, vocab_size):
        for word in list(vectors.stoi.keys())[:vocab_size]:
            if word in vocab.stoi:
                vocab.itos.append(word)
                vocab.stoi[word] = len(vocab.itos) - 1


def test_data_loader():
    from config import data_fields, train_file, valid_file, batch_size, device, skip_header, delimiter, pre_embeddings, \
        vector_cache, min_freq, extend_vocab, pre_vocab_size

    data_loader = DataLoader(data_fields, train_file, valid_file, batch_size, device, skip_header, delimiter,
                             pre_embeddings, vector_cache, min_freq, extend_vocab, pre_vocab_size)

    for x, y, length in data_loader.train_wrapper:
        print(length)
        print(x)
        print(y)
        break


if __name__ == '__main__':
    test_data_loader()
