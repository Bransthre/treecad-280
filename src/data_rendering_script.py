import vtkmodules
from cadquery import *
from cadquery.vis import show

import re
import os
import json
import tqdm

dataset_dir = "../cad-recode-v1.5/train/"
batch_ids = [f"0{i}" for i in range(10)] + list(range(10, 100))


def render_and_store_cad_file(batch_id, file_id, camera_rolls, camera_elevations):
    r = None
    with open(f"../cad-recode-v1.5/train/{batch_id}/{file_id}.py", "r") as f:
        cad = f.read()
    exec(cad)
    for img_idx, (roll, elevation) in enumerate(zip(camera_rolls, camera_elevations)):
        show(
            r,
            screenshot=f"../cad-recode-v1.5/train_img/{batch_id}/{file_id}-img_{img_idx}.png",
            roll=roll,
            elevation=elevation,
            interact=False,
        )


for batch_id in tqdm.tqdm(batch_ids, desc="Processing batches", leave=False):
    batch_dir = os.path.join(dataset_dir, f"batch_{batch_id}")
    vocabulary_pattern = r"\b\w+\s*\(.*?\)"
    all_vocabs_in_batch = {}

    # Check if the directory exists
    if os.path.exists(batch_dir):
        # Iterate through all files in the batch directory
        for _, file_name in tqdm.tqdm(
            enumerate(os.listdir(batch_dir)), desc=f"Processing files in {batch_dir}"
        ):
            file_path = os.path.join(batch_dir, file_name)
            if os.path.isfile(file_path):
                render_and_store_cad_file(
                    batch_id,
                    file_name.split(".")[0],
                    camera_rolls=[0, 90, 180, 270],
                    camera_elevations=[0, 45, 90, 135],
                )
    else:
        print(f"Directory {batch_dir} does not exist.")
