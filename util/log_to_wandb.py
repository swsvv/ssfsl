from pathlib import Path

import pandas as pd
import wandb

from config import Config


def log_to_wandb(config: Config, mode: str) -> None:
    print(f"📊 Log {config.exp_name} to WandB!")

    if mode == "train":
        run_name = f"train_{config.exp_name}"
    elif mode == "test":
        run_name = f"eval_{config.exp_name}"
    else:
        raise ValueError("Unknown mode!")

    wandb_config = {
        "project": config.project,
        "name": run_name,
        "config": config.model_dump(),
        "group": config.group,
        "reinit": True,
    }

    run = wandb.init(**wandb_config)

    config.wandb_run_id = run.id

    df = pd.read_csv(Path(config.exp_path, f"exp_{mode}.csv"))

    for _, row in df.iterrows():
        wandb.log(row.to_dict())

    run.finish()


if __name__ == "__main__":
    config = Config.parse_args(require_default_file=True)

    log_to_wandb(config, "train")
