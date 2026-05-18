from pathlib import Path

import csv


def save_test_result(
    exp_name: str,
    eval_dataset: str,
    num_way: int,
    num_spt: int,
    num_qry: int,
    ckpt_name: str,
    acc: float,
    std: float,
) -> None:
    dir_path = Path("~").expanduser() / "result"

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)
        print("Directory created successfully!")

    file_path = dir_path / Path(f"eval_{exp_name}.csv")
    columns = ["eval_dataset", "way", "shot", "query", "top_ckpt", "top_acc", "std"]
    result = [eval_dataset, num_way, num_spt, num_qry, ckpt_name, acc, std]

    try:
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0 and columns:
                writer.writerow(columns)
            writer.writerow(result)

    except IOError as e:
        print(f"Failed to write to file: {e}")


if __name__ == "__main__":
    save_test_result("TESTEXP", "TESTDATASET", 5, 5, 15, "CKPT-TEST", 0.9999, 0.9999)
