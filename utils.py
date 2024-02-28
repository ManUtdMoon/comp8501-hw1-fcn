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
            overall accuracy
            mean accuracy
            mean IU
            fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for p, l in zip(pred, label):
        # p, l.shape = (H,W)
        hist += _fast_hist(p.flatten(), l.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {
        "overall_acc": acc,
        "mean_acc": acc_cls,
        "freqw_acc": fwavacc,
        "mean_iu": mean_iu,
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



PALETTE = {
    '0': [0, 0, 0],
    '1': [128, 0, 0],
    '2': [0, 128, 0],
    '3': [128, 128, 0],
    '4': [0, 0, 128],
    '5': [128, 0, 128],
    '6': [0, 128, 128],
    '7': [128, 128, 128],
    '8': [64, 0, 0],
    '9': [192, 0, 0],
    '10': [64, 128, 0],
    '11': [192, 128, 0],
    '12': [64, 0, 128],
    '13': [192, 0, 128],
    '14': [64, 128, 128],
    '15': [192, 128, 128],
    '16': [0, 64, 0],
    '17': [128, 64, 0],
    '18': [0, 192, 0],
    '19': [128, 192, 0],
    '20': [0, 64, 128],
    '21': [128, 64, 128],
    '22': [0, 192, 128],
    '23': [128, 192, 128],
    '24': [64, 64, 0],
    '25': [192, 64, 0],
    '26': [64, 192, 0],
    '27': [192, 192, 0],
    '28': [64, 64, 128],
    '29': [192, 64, 128],
    '30': [64, 192, 128],
    '31': [192, 192, 128],
    '32': [0, 0, 64],
    '33': [128, 0, 64],
    '34': [0, 128, 64],
    '35': [128, 128, 64],
    '36': [0, 0, 192],
    '37': [128, 0, 192],
    '38': [0, 128, 192],
    '39': [128, 128, 192],
    '40': [64, 0, 64],
    '41': [192, 0, 64],
    '42': [64, 128, 64],
    '43': [192, 128, 64],
    '44': [64, 0, 192],
    '45': [192, 0, 192],
    '46': [64, 128, 192],
    '47': [192, 128, 192],
    '48': [0, 64, 64],
    '49': [128, 64, 64],
    '50': [0, 192, 64],
    '51': [128, 192, 64],
    '52': [0, 64, 192],
    '53': [128, 64, 192],
    '54': [0, 192, 192],
    '55': [128, 192, 192],
    '56': [64, 64, 64],
    '57': [192, 64, 64],
    '58': [64, 192, 64],
    '59': [192, 192, 64],
    '60': [64, 64, 192],
    '61': [192, 64, 192],
    '62': [64, 192, 192],
    '63': [192, 192, 192],
    '64': [32, 0, 0],
    '65': [160, 0, 0],
    '66': [32, 128, 0],
    '67': [160, 128, 0],
    '68': [32, 0, 128],
    '69': [160, 0, 128],
    '70': [32, 128, 128],
    '71': [160, 128, 128],
    '72': [96, 0, 0],
    '73': [224, 0, 0],
    '74': [96, 128, 0],
    '75': [224, 128, 0],
    '76': [96, 0, 128],
    '77': [224, 0, 128],
    '78': [96, 128, 128],
    '79': [224, 128, 128],
    '80': [32, 64, 0],
    '81': [160, 64, 0],
    '82': [32, 192, 0],
    '83': [160, 192, 0],
    '84': [32, 64, 128],
    '85': [160, 64, 128],
    '86': [32, 192, 128],
    '87': [160, 192, 128],
    '88': [96, 64, 0],
    '89': [224, 64, 0],
    '90': [96, 192, 0],
    '91': [224, 192, 0],
    '92': [96, 64, 128],
    '93': [224, 64, 128],
    '94': [96, 192, 128],
    '95': [224, 192, 128],
    '96': [32, 0, 64],
    '97': [160, 0, 64],
    '98': [32, 128, 64],
    '99': [160, 128, 64],
    '100': [32, 0, 192],
    '101': [160, 0, 192],
    '102': [32, 128, 192],
    '103': [160, 128, 192],
    '104': [96, 0, 64],
    '105': [224, 0, 64],
    '106': [96, 128, 64],
    '107': [224, 128, 64],
    '108': [96, 0, 192],
    '109': [224, 0, 192],
    '110': [96, 128, 192],
    '111': [224, 128, 192],
    '112': [32, 64, 64],
    '113': [160, 64, 64],
    '114': [32, 192, 64],
    '115': [160, 192, 64],
    '116': [32, 64, 192],
    '117': [160, 64, 192],
    '118': [32, 192, 192],
    '119': [160, 192, 192],
    '120': [96, 64, 64],
    '121': [224, 64, 64],
    '122': [96, 192, 64],
    '123': [224, 192, 64],
    '124': [96, 64, 192],
    '125': [224, 64, 192],
    '126': [96, 192, 192],
    '127': [224, 192, 192],
    '128': [0, 32, 0],
    '129': [128, 32, 0],
    '130': [0, 160, 0],
    '131': [128, 160, 0],
    '132': [0, 32, 128],
    '133': [128, 32, 128],
    '134': [0, 160, 128],
    '135': [128, 160, 128],
    '136': [64, 32, 0],
    '137': [192, 32, 0],
    '138': [64, 160, 0],
    '139': [192, 160, 0],
    '140': [64, 32, 128],
    '141': [192, 32, 128],
    '142': [64, 160, 128],
    '143': [192, 160, 128],
    '144': [0, 96, 0],
    '145': [128, 96, 0],
    '146': [0, 224, 0],
    '147': [128, 224, 0],
    '148': [0, 96, 128],
    '149': [128, 96, 128],
    '150': [0, 224, 128],
    '151': [128, 224, 128],
    '152': [64, 96, 0],
    '153': [192, 96, 0],
    '154': [64, 224, 0],
    '155': [192, 224, 0],
    '156': [64, 96, 128],
    '157': [192, 96, 128],
    '158': [64, 224, 128],
    '159': [192, 224, 128],
    '160': [0, 32, 64],
    '161': [128, 32, 64],
    '162': [0, 160, 64],
    '163': [128, 160, 64],
    '164': [0, 32, 192],
    '165': [128, 32, 192],
    '166': [0, 160, 192],
    '167': [128, 160, 192],
    '168': [64, 32, 64],
    '169': [192, 32, 64],
    '170': [64, 160, 64],
    '171': [192, 160, 64],
    '172': [64, 32, 192],
    '173': [192, 32, 192],
    '174': [64, 160, 192],
    '175': [192, 160, 192],
    '176': [0, 96, 64],
    '177': [128, 96, 64],
    '178': [0, 224, 64],
    '179': [128, 224, 64],
    '180': [0, 96, 192],
    '181': [128, 96, 192],
    '182': [0, 224, 192],
    '183': [128, 224, 192],
    '184': [64, 96, 64],
    '185': [192, 96, 64],
    '186': [64, 224, 64],
    '187': [192, 224, 64],
    '188': [64, 96, 192],
    '189': [192, 96, 192],
    '190': [64, 224, 192],
    '191': [192, 224, 192],
    '192': [32, 32, 0],
    '193': [160, 32, 0],
    '194': [32, 160, 0],
    '195': [160, 160, 0],
    '196': [32, 32, 128],
    '197': [160, 32, 128],
    '198': [32, 160, 128],
    '199': [160, 160, 128],
    '200': [96, 32, 0],
    '201': [224, 32, 0],
    '202': [96, 160, 0],
    '203': [224, 160, 0],
    '204': [96, 32, 128],
    '205': [224, 32, 128],
    '206': [96, 160, 128],
    '207': [224, 160, 128],
    '208': [32, 96, 0],
    '209': [160, 96, 0],
    '210': [32, 224, 0],
    '211': [160, 224, 0],
    '212': [32, 96, 128],
    '213': [160, 96, 128],
    '214': [32, 224, 128],
    '215': [160, 224, 128],
    '216': [96, 96, 0],
    '217': [224, 96, 0],
    '218': [96, 224, 0],
    '219': [224, 224, 0],
    '220': [96, 96, 128],
    '221': [224, 96, 128],
    '222': [96, 224, 128],
    '223': [224, 224, 128],
    '224': [32, 32, 64],
    '225': [160, 32, 64],
    '226': [32, 160, 64],
    '227': [160, 160, 64],
    '228': [32, 32, 192],
    '229': [160, 32, 192],
    '230': [32, 160, 192],
    '231': [160, 160, 192],
    '232': [96, 32, 64],
    '233': [224, 32, 64],
    '234': [96, 160, 64],
    '235': [224, 160, 64],
    '236': [96, 32, 192],
    '237': [224, 32, 192],
    '238': [96, 160, 192],
    '239': [224, 160, 192],
    '240': [32, 96, 64],
    '241': [160, 96, 64],
    '242': [32, 224, 64],
    '243': [160, 224, 64],
    '244': [32, 96, 192],
    '245': [160, 96, 192],
    '246': [32, 224, 192],
    '247': [160, 224, 192],
    '248': [96, 96, 64],
    '249': [224, 96, 64],
    '250': [96, 224, 64],
    '251': [224, 224, 64],
    '252': [96, 96, 192],
    '253': [224, 96, 192],
    '254': [96, 224, 192]
}

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