from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
import tqdm
import argparse


def main(args):
    pipe = pipeline(task="depth-estimation", 
                    model="depth-anything/Depth-Anything-V2-Small-hf", 
                    use_fast=True)
    paths = sorted(list(glob(os.path.join(args.input_path, '*.png'))))
    for path in tqdm.tqdm(paths):
        image = Image.open(path)
        depth = pipe(image)["depth"]
        save_path = os.path.dirname(path).replace('input', 'depth')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        depth.save(os.path.join(save_path, os.path.basename(path)))
        # plt.imshow(depth, cmap='viridis')
        # plt.axis('off')
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation Pipeline")
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input images directory')
    args = parser.parse_args()
    main(args)
