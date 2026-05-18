import csv
import os
import shutil
import argparse


def prepare_dataset(args):
    split_list = ["train", "val", "test"]
    for split in split_list:
        dataset_split_path = os.path.join(args.dataset_path, args.dataset, split)

        if not os.path.exists(dataset_split_path):
            os.makedirs(dataset_split_path)

        csv_file_path = os.path.join(args.dataset_csv_path, f"{split}.csv")
        with open(csv_file_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)

            next(csv_reader)

            for row in csv_reader:
                folder_path = row[1]
                destination_path = os.path.join(dataset_split_path, folder_path)

                source_path = os.path.join(args.dataset_path, args.dataset, folder_path)

                if os.path.exists(source_path):
                    shutil.move(source_path, destination_path)
                    print(f"Moved {folder_path} to {destination_path}")
                else:
                    print(f"Folder {folder_path} does not exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/path/to/dataset/",
        help="datset path",
    )
    parser.add_argument(
        "--dataset_csv_path",
        type=str,
        default="./miniImageNet/",
        help="datset info file path",
    )
    parser.add_argument("--dataset", type=str, default="miniImageNet", help="datset")

    args = parser.parse_args()
    prepare_dataset(args)
