from pathlib import Path
import shutil
import argparse
import pandas as pd
import numpy as np


def create_isic(csv_path: str, source_dir: str, destination_dir: str) -> None:
    """adapted from https://github.com/IBM/cdfsl-benchmark/blob/master/datasets/ISIC_few_shot.py"""
    data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

    # First column contains the image paths
    image_names = np.asarray(data_info.iloc[:, 0])

    labels = np.asarray(data_info.iloc[:, 1:])
    labels = (labels != 0).argmax(axis=1)

    # Define paths
    source_dir = Path(source_dir)
    csv_path = Path(csv_path)
    destination_dir = Path(destination_dir)

    # Create the destination base directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Traverse the root directory and all subdirectories to find .jpg files
    for jpg_file in source_dir.rglob("*.jpg"):
        if jpg_file.stem in image_names:
            index = np.where(image_names == jpg_file.stem)[0]
            label = labels[index]

            # Define the full path for the destination label directory
            label_dir = destination_dir / label
            # Create the label directory if it doesn't exist
            label_dir.mkdir(parents=True, exist_ok=True)
            # Define the full path for the destination file
            dest_file = label_dir / jpg_file.name
            # Copy the file
            shutil.copy(jpg_file, dest_file)
            print(f"Copied: {jpg_file} to {dest_file}")

    print("All .jpg files have been copied successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/path/to/ISIC/ISIC2018_Task3_Training_Input",
        help="Original data path",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/path/to/dataset/CDFSL_raw/ISIC/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
        help="The path that preprocessed ChestX csv file is located",
    )
    parser.add_argument(
        "--destination_dir",
        type=str,
        default="/path/to/dataset/CDFSL/ISIC",
        help="Selected data path",
    )
    args = parser.parse_args()

    create_isic(args.csv_path, args.source_dir, args.destination_dir)
