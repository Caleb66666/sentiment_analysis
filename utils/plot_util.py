import matplotlib.pyplot as plt

# 无图形界面需要加，否则plt报错
plt.switch_backend('MacOSX')


def loss_acc_plot(history, plot_path):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    train_accuracy = history['train_acc']
    eval_accuracy = history['eval_acc']

    fig = plt.figure(figsize=(12, 6))
    fig.add_subplot(2, 1, 1)
    plt.title('loss during train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(2, 1, 2)
    plt.title('accuracy during train')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_accuracy)
    plt.plot(epochs, eval_accuracy)
    plt.legend(['train_acc', 'eval_acc'])

    plt.subplots_adjust(wspace=0, hspace=0.45)  # 调整子图间距

    plt.savefig(plot_path)


if __name__ == '__main__':
    history_ = {
        'train_loss': range(0, 99, 1),
        'eval_loss': range(5, 104, 1),
        'train_acc': range(5, 104, 1),
        'eval_acc': range(0, 99, 1)
    }
    loss_acc_plot(history_, plot_path="./test")
