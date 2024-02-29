"""Usage
python predict_reference.py -o data/outputs/
"""

import os
import numpy as np
import pathlib
import click
from matplotlib import pyplot as plt
import torch
import tqdm

from dataset import get_dataset
from utils import PALETTE, collate_fn
from predict import _ind_to_rgb

@click.command()
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
def main(output_dir, device):
    # create output_dir
    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir = output_dir / "torchvision"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # get dataset
    is_train = True
    is_val = False
    is_test = (not is_train) and (not is_val)
    bz = 1 if is_test else 16
    dataset = get_dataset(is_train=is_train, is_eval=is_val)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bz, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    # predict
    with torch.inference_mode():
        losses = list()
        image_list = list()
        pred_list = list()
        label_list = list()
        
        with tqdm.tqdm(dataloader, desc="Predicting",
            leave=False, mininterval=1
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # device transfer
                image, mask = batch
                image, mask = image.to(device), mask.to(device)
                pred = model(image)["out"]
                
                image_list.append(image.cpu().numpy())
                pred_list.append(pred.argmax(1).cpu().numpy())
                label_list.append(mask.cpu().numpy())

                if batch_idx == 1:
                    break

    # save image, pred, label into one image to output_dir
    for i, pair in enumerate(zip(image_list, pred_list, label_list)):
        # image: (B, 3, H, W), [0, 1] rgb
        # pred: (B, H, W), label: (B, H, W), [0, num_classes-1]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image, pred, label = pair

        for j in range(bz):        
            this_image = np.clip(
                image[j].transpose(1, 2, 0) * std + mean,
                0.0, 1.0
            )
            this_pred = _ind_to_rgb(pred[j].astype(np.uint8))
            this_label = _ind_to_rgb(label[j].astype(np.uint8))

            f, axarr = plt.subplots(1, 3)
            axarr[0].set_title("Image")
            axarr[0].imshow(this_image)

            axarr[1].set_title("Pred")
            axarr[1].imshow(this_pred)

            axarr[2].set_title("Label")
            axarr[2].imshow(this_label)

            f.tight_layout()
            f.savefig(output_dir / f"{i*bz+j}.png", dpi=150)

            # close the figure
            plt.close(f)


if __name__ == "__main__":
    main()