import gc
from cadquery import *
from naive_autoregressive.no_interaction_vis import no_interact_show
import cadquery as cq
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMG_H = 128
IMG_W = 128


def load_cad_data(cad_path):
    """
    loads and parses cadquery data from .py path specified
    """
    with open(cad_path, "r") as f:
        cad_text = f.read()
    cad_data = {"r": None}
    exec(cad_text, cad_data)
    return cad_data["r"]


def render_cad_data(cad_data, rolls, elevations):
    """
    renders cad_data (cadquery workplane object) for given rolls and elevations
    returns list of images
    """
    images = []
    for roll, elevation in zip(rolls, elevations):
        img = no_interact_show(
            cad_data,
            roll=roll,
            elevation=elevation,
            interact=False,
        )
        images.append(img)
    return images


def main(
    dataset_dir,
    train_size,
    val_size,
    output_dir,
    train_start=0,
    images_per_cad=8,
    save_every=1000,
):
    """
    given dataset directory, size of training and validation split
    saves renders to output dir
    """
    for split, data_limit in [("train", train_size), ("val", val_size)]:
        split_path = os.path.join(dataset_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        split_images = []
        split_rolls = []
        split_elevations = []
        split_cad_indices = []

        for i in tqdm(
            range(train_start, train_start + data_limit),
            initial=train_start,
            total=train_start + data_limit,
        ):
            # get cad file paths
            filename = f"{i}.py"
            if split == "train":
                batch_idx = i // 10000
                data_path = os.path.join(split_path, f"batch_{batch_idx:02}", filename)
            else:
                data_path = os.path.join(split_path, filename)

            # read and parse cad files
            if os.path.exists(data_path):
                cad_data = load_cad_data(data_path)
                # render images
                rolls = np.random.uniform(
                    low=-180.0, high=180.0, size=(images_per_cad,)
                )
                elevations = np.random.uniform(
                    low=-90.0, high=90.0, size=(images_per_cad,)
                )
                cad_images = render_cad_data(
                    cad_data, rolls, elevations
                )  # list(H, W, C)
                cad_images = np.stack(cad_images, axis=0)  # (P, H, W, C)
                cad_index = i

            else:
                continue

            split_images.append(cad_images)
            split_rolls.append(rolls)
            split_elevations.append(elevations)
            split_cad_indices.append(cad_index)

            if (len(split_images) >= save_every) or i + 1 == data_limit:
                split_images_stack = np.stack(split_images, axis=0)  # (B, P, H, W, C)
                split_rolls_stack = np.stack(split_rolls, axis=0)  # (B, P)
                split_elevations_stack = np.stack(split_elevations, axis=0)  # (B, P)
                split_cad_indices_stack = np.stack(split_cad_indices, axis=0)
                out_path = os.path.join(split_output_dir, f"{i+1:05}")
                np.save(f"{out_path}_images", split_images_stack)
                np.save(f"{out_path}_rolls", split_rolls_stack)
                np.save(f"{out_path}_elevations", split_elevations_stack)
                np.save(f"{out_path}_cad_indices", split_cad_indices_stack)
                split_images, split_rolls, split_elevations = [], [], []
                print(f"[{i+1} of {data_limit}] saved batch to {out_path}")


if __name__ == "__main__":
    dataset_dir = "datasets/cad-recode-v1.5"
    output_dir = "datasets/cad-recode-render-v1"
    train_size = 10000
    val_size = 0
    images_per_cad = 8

    main(
        dataset_dir,
        train_size,
        val_size,
        output_dir,
        train_start=80000,
        images_per_cad=images_per_cad,
    )
