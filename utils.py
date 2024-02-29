from typing import Dict, Callable, Optional
import os
import numpy as np
import torch
import csv
from datetime import datetime
import pathlib
from pathlib import Path


PROJ_DIR = str(pathlib.Path(__file__).parent.expanduser())


def _fast_hist(pred, label, n_class):
    mask = np.logical_and(label >= 0, label < n_class)
    pred_legal = np.logical_and(pred >= 0, pred < n_class)
    assert np.all(pred_legal), "Illegal prediction value"
    hist = np.bincount(
        n_class * label[mask].astype(int) + pred[mask],
        minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist

def eval_score(pred, label, n_class):
    """Returns accuracy score evaluation result.

    @params
        pred: (B,H,W) ndarray, predicted label
        label: (B,H,W) ndarray, ground truth label
        n_class: int, number of classes
    @return
        result: dict
            global accuracy
            mean accuracy
            mean IU
    """
    hist = np.zeros((n_class, n_class), dtype=np.int64)
    for p, l in zip(pred, label):
        # p, l.shape = (H,W)
        hist += _fast_hist(p.flatten(), l.flatten(), n_class)
    h = hist.astype(np.float64)

    acc_global = np.diag(h).sum() / h.sum()

    acc = np.diag(h) / h.sum(1)

    mean_iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))

    return {
        "acc_global": acc_global,
        "acc": acc.mean(),
        "mean_iu": mean_iu.mean(),
    }

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


class Log:
    def __init__(self):
        self._dir = None
        self._log_file = None

    def setup(self, dir, log_filename='log.txt', if_exists='append'):
        # assert self._dir is None, 'Can only setup once'
        self._dir = Path(dir)
        self._dir.mkdir(exist_ok=True)
        print(f'Set log dir to {dir}, log filename to {log_filename}')
        log_path = self._dir / log_filename
        if log_path.exists():
            if if_exists == 'append':
                print(f'Log file named {log_filename} already exists; will use append mode')
                self._log_file = log_path.open('a', buffering=1)
            elif if_exists == 'overwrite':
                print(f'Log file named {log_filename} already exists; will overwrite it')
                self._log_file = log_path.open('w', buffering=1)
            elif if_exists == 'exit':
                print(f'Log file named {log_filename} already exists; exiting')
                exit()
            else:
                raise NotImplementedError(f'Unknown if_exists option: {if_exists}')
        else:
            print(f'Creating new log file named {log_filename}')
            self._log_file = log_path.open('w', buffering=1)

    @property
    def dir(self):
        return self._dir

    def message(self, message, timestamp=True, flush=False):
        if timestamp:
            now_str = datetime.now().strftime('%H:%M:%S')
            message = f'[{now_str}] ' + message
        else:
            message = ' ' * 11 + message
        print(message)
        self._log_file.write(f'{message}\n')
        if flush:
            self._log_file.flush()

    def __call__(self, *args, **kwargs):
        return self.message(*args, **kwargs)

default_log = Log()


class TabularLog:
    def __init__(self, dir, filename):
        self._dir = Path(dir)
        assert self._dir.is_dir()
        self._filename = filename
        self._column_names = None
        self._file = open(self.path, mode=('a' if self.path.exists() else 'w'), newline='')
        self._writer = csv.writer(self._file)

    @property
    def path(self):
        return self._dir/self._filename

    def row(self, row, flush=True):
        if self._column_names is None:
            self._column_names = list(row.keys())
            self._writer.writerow(self._column_names)
        self._writer.writerow([row[col] for col in self._column_names])
        if flush:
            self._file.flush()


class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path



PALETTE = np.asarray(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
)

CLASSES = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
    "tv/monitor": 20,
}