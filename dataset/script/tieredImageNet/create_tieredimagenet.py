from pathlib import Path
import csv
import shutil
import argparse


def create_tiered_imagenet(args):
    # split_list = ["train", "val", "test"]
    split_list = ["test"]

    for split in split_list:
        if split == "val":
            source_split_path = Path(args.dataset_path, args.source_name, "train")
        elif split == "test":
            source_split_path = Path(args.dataset_path, args.source_name, "train")

        if not source_split_path.is_dir():
            raise ValueError(f"There is no {args.source_name}!")

        target_split_path = Path(args.dataset_path, args.target_name, split)
        target_split_path.mkdir(parents=True, exist_ok=True)

        csv_file_path = Path(args.target_csv_path, f"{split}.csv")

        with open(csv_file_path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)

            # if there is a header skip header in csv
            next(csv_reader)

            for row in csv_reader:
                # row's order is depend on dataset csv
                file_name = row[0]
                class_name = row[1]

                # source_class_path = Path(source_split_path, class_name)
                source_class_path = Path(source_split_path)
                target_class_path = Path(target_split_path, class_name)
                target_class_path.mkdir(parents=True, exist_ok=True)

                if source_class_path.is_dir():
                    file_name = file_name.replace("\\", "/")
                    source_file_path = Path(source_class_path, file_name)
                    target_file_path = Path(target_split_path, file_name)
                    shutil.copy(str(source_file_path), str(target_file_path))
                    print(f"Copy {file_name} ----> {target_class_path}")
                else:
                    print(f"There is no {file_name}!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="~/common_dataset/",
        help="The dataset path",
    )
    parser.add_argument(
        "--source_name",
        type=str,
        default="imagenet2012",
        help="The source datset name",
    )
    parser.add_argument(
        "--target_csv_path",
        type=str,
        default="./tieredImageNet/",
        help="The path that target dataset csv files are located",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="tieredImageNet",
        help="The target datset name to create",
    )

    args = parser.parse_args()

    create_tiered_imagenet(args)
