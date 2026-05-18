from pathlib import Path
import argparse
import shutil
import csv
import numpy as np
import pandas as pd


def create_chestx_csv(csv_path: str) -> None:
    """Adapted from https://github.com/IBM/cdfsl-benchmark/blob/master/datasets/Chest_few_shot.py"""
    used_labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
    ]

    labels_maps = {
        "Atelectasis": 0,
        "Cardiomegaly": 1,
        "Effusion": 2,
        "Infiltration": 3,
        "Mass": 4,
        "Nodule": 5,
        "Pneumothorax": 6,
    }

    data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

    image_name_all = np.asarray(data_info.iloc[:, 0])
    labels_all = np.asarray(data_info.iloc[:, 1])

    csv_data = [["Image Index", "Finding Labels"]]

    for name, label in zip(image_name_all, labels_all):
        label = label.split("|")

        if (
            len(label) == 1
            and label[0] != "No Finding"
            and label[0] != "Pneumonia"
            and label[0] in used_labels
        ):
            csv_data.append([name, labels_maps[label[0]]])

    with open("cdfsl_chestx.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)


def create_chestx(root_dir: str, csv_file_path: str, destination_base_dir: str) -> None:
    # Define paths
    root_dir = Path(root_dir)
    csv_file_path = Path(csv_file_path)
    destination_base_dir = Path(destination_base_dir)

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Create the destination base directory if it doesn't exist
    destination_base_dir.mkdir(parents=True, exist_ok=True)

    # Create a dictionary to map image names to labels
    image_label_map = dict(zip(df["Image Index"], df["Finding Labels"]))

    # Traverse the root directory and all subdirectories to find .png files
    for current_dir in root_dir.rglob("*"):
        if current_dir.is_dir():
            for inside_dir in current_dir.rglob("*"):
                if inside_dir.is_dir():
                    # Find all .png files in the current directory
                    png_files = inside_dir.glob("*.png")

                    # Copy each .png file to the respective label directory
                    for png_file in png_files:
                        image_name = png_file.name
                        if image_name in image_label_map:
                            label = image_label_map[image_name]
                            # Define the full path for the destination label directory
                            label_dir = destination_base_dir / str(label)
                            # Create the label directory if it doesn't exist
                            label_dir.mkdir(parents=True, exist_ok=True)
                            # Define the full path for the destination file
                            dest_file = label_dir / image_name
                            # Copy the file
                            shutil.copy(png_file, dest_file)
                            print(f"Copied: {png_file} to {dest_file}")
                else:
                    print("Is it correct directory?")

    print("All .png files have been copied successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_csv_path",
        type=str,
        default="/path/to/dataset/CDFSL_raw/ChestX/Data_Entry_2017.csv",
        help="The path that target dataset csv files are located",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="./cdfsl_chestx.csv",
        help="The path that preprocessed ChestX csv file is located",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/path/to/ChestX/",
        help="Original data path",
    )
    parser.add_argument(
        "--destination_base_dir",
        type=str,
        default="/path/to/dataset/CDFSL/ChestX",
        help="Selected data path",
    )
    args = parser.parse_args()

    create_chestx_csv(args.target_csv_path)
    create_chestx(args.root_dir, args.csv_file_path, args.destination_base_dir)
