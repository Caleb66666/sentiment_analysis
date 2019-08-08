import torch
from torch import nn
from torch.nn import functional as f
from utils.ml_util import embed_init_unk


class TextCnn(nn.Module):
    def __init__(self, vocab, embed_dim, n_filters, filter_sizes, num_labels, num_classes, dropout, n_hidden, unk_idx,
                 pad_idx, use_pre_embed):
        super(TextCnn, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if use_pre_embed:
            self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=pad_idx)
            self.embedding.weight.data.copy_(vocab.vectors)
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=pad_idx)
            self.embedding.weight.data[unk_idx] = embed_init_unk(len(vocab), embed_dim)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim)),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        ) for fs in filter_sizes])

        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(n_hidden, num_labels * num_classes),
        )

        self.num_classes = num_classes
        self.num_labels = num_labels

    def forward(self, inputs):
        embedded = self.embedding(inputs).unsqueeze(1)
        conveds = [conv(embedded) for conv in self.convs]
        pooleds = [f.max_pool1d(conved.squeeze(3), conved.shape[2]).squeeze() for conved in conveds]
        outputs = self.dropout(torch.cat(tuple(pooleds), dim=1))
        logits = self.fc(outputs)
        logits = logits.view((-1, self.num_classes, self.num_labels))
        return logits
