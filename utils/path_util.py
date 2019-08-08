import os
import dill
import torch


def fetch_project_path(symbol="utils"):
    cur_dir = os.path.dirname(__file__)
    project_path = ""
    while len(cur_dir) > 1:
        if symbol in os.listdir(cur_dir):
            project_path = cur_dir
            break
        cur_dir = os.path.dirname(cur_dir)
    return project_path


pro_path = fetch_project_path()


def abspath(relative_file, parent_path=pro_path):
    if parent_path not in relative_file:
        return os.path.join(parent_path, relative_file)
    return relative_file


def local_save(save_path, py_obj):
    with open(save_path, "wb") as f:
        dill.dump(py_obj, f)


def local_load(load_path):
    with open(load_path, "rb") as f:
        py_obj = dill.load(f)
    return py_obj


def save_checkpoint(ckpt_file, cur_epoch, model, optimizer):
    torch.save({
        "cur_epoch": cur_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, ckpt_file)


def load_checkpoint(ckpt_file):
    checkpoint = torch.load(ckpt_file)
    return checkpoint


if __name__ == '__main__':
    print(pro_path)
