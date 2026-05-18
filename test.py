from pathlib import Path

from tqdm.rich import tqdm

import torch

from config import Config
from logger import Logger
from dataloader import get_dataloader
from util.get_module import get_encoder, get_model
from util.stats import mean_confidence_interval
from util.ckpt import get_ckpt_list
from util.set_seed import set_seed
from util.save_test_result import save_test_result
from util.log_to_wandb import log_to_wandb


@torch.no_grad()
def test(config: Config, ckpt: str) -> tuple[float, float]:
    ckpt_path = Path(config.exp_path, ckpt)
    test_loader = get_dataloader(config, "test")

    encoder = get_encoder(config.encoder)
    encoder = encoder.to(config.device)

    ckpt_state_dict = torch.load(ckpt_path, map_location=config.device)
    encoder.load_state_dict(ckpt_state_dict["encoder_state_dict"])

    model = get_model("EpisodeTest", config, encoder)
    model = model.to(config.device)
    model.eval()

    accuracies = []
    for _ in tqdm(
        range(config.num_episode),
        desc=f"CKPT: {ckpt}, Test ({model.n_way}-way, {model.num_spt}-shot, {model.num_qry}-qry)",
        position=0,
        leave=False,
    ):
        for x, y in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            acc = model(x, y)
            accuracies.append(acc)

    ckpt_acc, ckpt_std = mean_confidence_interval(accuracies)

    return ckpt_acc, ckpt_std


def test_ckpts(config: Config) -> None:
    logger = Logger(file_path=config.exp_path, mode=config.mode)

    top_acc = 0.0
    top_std = 0.0
    top_ckpt = ""

    ckpt_list = get_ckpt_list(config.exp_path)

    for idx, ckpt in enumerate(ckpt_list):
        ckpt_acc, ckpt_std = test(config, ckpt)

        logger.record("test/step", idx)
        logger.record(
            f"test/acc_{config.n_way}-way_{config.num_spt}-shot",
            ckpt_acc,
        )
        logger.record(
            f"test/std_{config.n_way}-way_{config.num_spt}-shot",
            ckpt_std,
        )
        logger.info(
            f"ckpt: {ckpt}, {config.n_way}-way, {config.num_spt}-shot, acc: {ckpt_acc:.6f}, std: {ckpt_std:.6f}"
        )

        logger.dump(step=idx)

        if ckpt_acc > top_acc:
            top_acc = ckpt_acc
            top_std = ckpt_std
            top_ckpt = ckpt

            logger.info(
                f"✨ Best acc is updated! --- {ckpt}, acc: {ckpt_acc:.6f}, std: {ckpt_std:.6f}"
            )

    logger.info(f"🌋 Best acc: {top_acc:.6f} at {top_ckpt}")

    save_test_result(
        config.exp_name,
        config.eval_dataset,
        config.n_way,
        config.num_spt,
        config.num_qry,
        top_ckpt,
        top_acc,
        top_std,
    )

    log_to_wandb(config, config.mode)


if __name__ == "__main__":
    config = Config.parse_args(require_default_file=True)

    if config.mode != "test":
        print(f"⚠️  Config's mode is {config.mode} -> change to 'test'!")
        config.mode = "test"

    set_seed(config.seed)

    test_ckpts(config)
