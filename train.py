from model import TextCnn
from utils.logger_util import MyLogger
from utils.path_util import save_checkpoint, load_checkpoint
from utils.ml_util import init_network, adjust_lr
from utils.measure_util import f1_measure
from data_loader import DataLoader
from torch import nn, optim
from tqdm import tqdm
from config import *
import torch
import os
import time


def train(model, batches, optimizer, criterion, classes):
    epoch_loss, epoch_acc = 0.0, 0.0
    model.train()

    for x, y, lengths in tqdm(batches):
        logits = model(x)
        loss = criterion(logits, y)
        acc = f1_measure(logits, y, classes)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(batches), epoch_acc / len(batches)


def evaluate(model, batches, criterion, classes):
    epoch_loss, epoch_acc = 0.0, 0.0
    model.eval()

    with torch.no_grad():
        for x, y, lengths in tqdm(batches):
            logits = model(x)
            loss = criterion(logits, y)
            acc = f1_measure(logits, y, classes)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(batches), epoch_acc / len(batches)


def main():
    logger = MyLogger(task_name, log_file, summary_dir)
    logger.info(f"task name: {task_name}, device: {device}")

    dl = DataLoader(data_fields, train_file, valid_file, batch_size, device, skip_header, delimiter, pre_embeddings,
                    vector_cache, min_freq, extend_vocab, pre_vocab_size)
    logger.info(f"train len: {len(dl.train_ds)}, valid len: {len(dl.valid_ds)}, train batches: {len(dl.train_wrapper)},"
                f"valid batches: {len(dl.valid_wrapper)}, batch size: {batch_size}")

    model = TextCnn(dl.vocab, embed_dim, n_filters, filter_sizes, dl.num_labels, dl.num_classes, dropout, n_hidden,
                    dl.unk_idx, dl.pad_idx)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=weight_decay)
    if os.path.exists(model_file) and resume:
        checkpoint = load_checkpoint(model_file)
        cur_epoch = checkpoint["cur_epoch"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(f"load trained model: {task_name}, epoch: {cur_epoch}")
    else:
        cur_epoch = 0
        init_network(model)

    correspond_loss = 0.0
    correspond_epoch = 0
    best_f1 = 0.0
    for epoch in range(cur_epoch, epochs):
        cur_lr = adjust_lr(optimizer, epoch, init_lr, lr_decay, min_lr)
        tic = time.time()
        train_loss, train_acc = train(model, dl.train_wrapper, optimizer, criterion, dl.classes)
        valid_loss, valid_acc = evaluate(model, dl.valid_wrapper, criterion, dl.classes)
        toc = time.time()

        logger.info(f"Epoch: {epoch + 1:02}/{epochs}, costs: {toc - tic:.2f}(s), learning rate: {cur_lr:.5f}")
        logger.info(f"Train loss: {train_loss:.3f}, Train F1: {train_acc * 100:.2f}%")
        logger.info(f"Valid loss: {valid_loss:.3f}, Valid F1: {valid_acc * 100:.2f}%\n")
        logger.add_scalar("train/loss", train_loss, epoch + 1)
        logger.add_scalar("train/f1", train_acc, epoch + 1)
        logger.add_scalar("train/lr", cur_lr, epoch + 1)
        logger.add_scalar("valid/loss", valid_loss, epoch + 1)
        logger.add_scalar("valid/f1", valid_acc, epoch + 1)

        if best_f1 < valid_acc:
            best_f1 = valid_acc
            correspond_epoch = epoch + 1
            correspond_loss = valid_loss

    logger.info(f"Training completes. best f1: {best_f1 * 100:.2f}%, loss: {correspond_loss:.3f}, "
                f"epoch: {correspond_epoch}\n")
    save_checkpoint(model_file, epochs - 1, model, optimizer)


if __name__ == '__main__':
    main()
