import shutil
import argparse
from pathlib import Path


def prepare_dataset(args):
    split_list = ["train", "val", "test"]
    for split in split_list:
        dataset_split_path = Path(args.dataset_root, split)

        if not dataset_split_path.exists():
            dataset_split_path.mkdir()

        file_path = f"./{split}.txt"
        with open(file_path, "rb") as file:
            for row in file:
                row = row.rstrip(b"\r\n").decode("utf-8")
                destination_path = Path(dataset_split_path, row)
                source_path = Path(args.dataset_root, row)

                if source_path.exists():
                    shutil.move(source_path, destination_path)
                    print(f"{row} was moved to ====> {destination_path}")
                else:
                    print(f"There is no {row}!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/path/to/dataset/Omniglot/",
        help="datset info file path",
    )

    args = parser.parse_args()
    prepare_dataset(args)
