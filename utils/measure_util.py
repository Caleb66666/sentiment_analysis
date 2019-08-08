import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def categorical_accuracy(predictions, y):
    max_dim = predictions.argmax(dim=1, keepdim=True)
    correct = max_dim.squeeze(1).eq(y)
    return correct.sum() / torch.scalar_tensor(y.shape[0])


def measure_accuracy(logits, y, arg_dim=1):
    """
    准确度
    :param logits: 经过softmax处理过的模型输出logits
    :param y:
    :param arg_dim:
    :return:

    y_true = torch.tensor([3, 0, 2]).unsqueeze(1)
    outputs = torch.tensor([[0.1, 0.2, 0.4, 0.7],
                            [0.9, 0.1, 0.1, 0.1],
                            [0.4, 0.7, 0.1, 0.1]]).float().unsqueeze(-1)
    outputs = outputs.log_softmax(dim=1)
    print(measure_accuracy(outputs, y_true))
    """
    logits = logits.argmax(dim=arg_dim, keepdim=True).squeeze(arg_dim)
    return accuracy_score(y, logits)


def f1_measure(logits, y_true, classes, arg_dim=1):
    mb = MultiLabelBinarizer(classes=classes)

    y_pred = logits.argmax(dim=arg_dim, keepdim=True).squeeze(arg_dim)
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    return f1_score(mb.fit_transform(y_true), mb.fit_transform(y_pred), average="macro")


if __name__ == '__main__':
    logits_ = torch.tensor([
        [[0.1, 0.7, 0.1],
         [0.7, 0.3, 0.5],
         [0.3, 0.1, 0.9]],
        [[0.9, 0.7, 0.1],
         [0.4, 0.3, 0.5],
         [0.3, 0.1, 0.9]],
        [[0.1, 0.7, 0.1],
         [0.7, 0.3, 0.5],
         [0.3, 0.1, 0.9]],
        [[0.1, 0.7, 0.1],
         [0.7, 0.3, 0.5],
         [0.3, 0.1, 0.9]],
        [[0.1, 0.2, 0.1],
         [0.7, 0.3, 0.5],
         [0.3, 0.1, 0.9]],
        [[0.1, 0.7, 0.1],
         [0.7, 0.3, 0.5],
         [0.3, 0.1, 0.9]],
    ])

    y_true_ = torch.tensor([
        [1, 0, 2],
        [1, 0, 2],
        [1, 0, 2],
        [1, 0, 2],
        [1, 0, 2],
        [1, 0, 2],
    ])

    classes_ = [0, 1, 2]
    print(f1_measure(logits_, y_true_, classes_))
