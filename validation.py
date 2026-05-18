import torch
from tqdm.rich import tqdm

from interface import UnSup, Sup
from config import Config
from logger import Logger

from dataloader import get_dataloader
from util.get_module import get_model


@torch.no_grad()
def validation_loss(config: Config, logger: Logger, model: UnSup | Sup) -> float:
    model.eval()

    val_loader = get_dataloader(config, "val", "loss")

    for x, _ in val_loader:
        x = [x_i.to(config.device) for x_i in x]

        val_loss, val_corr_diff = model(x)

        logger.record_mean("val/loss", val_loss.item())
        logger.record_mean("val/corr_diff", val_corr_diff.item())

    model.train()
    return logger.data["val/loss"]


@torch.no_grad()
def validation_acc(config: Config, logger: Logger, model: UnSup | Sup) -> float:
    model.eval()

    config.num_episode = 1000
    config.n_way = 5
    config.num_spt = 5
    config.num_qry = 15
    config.classifier = "LogisticRegression"

    encoder = model.encoder

    val_loader = get_dataloader(config, "val", "acc")
    val_model = get_model("EpisodeTest", config, encoder)
    val_model = val_model.to(config.device)
    val_model.eval()

    for episode in tqdm(
        range(config.num_episode),
        desc=f"Validation ({val_model.n_way}-way, {val_model.num_spt}-shot, {val_model.num_qry}-qry)",
        position=0,
        leave=False,
    ):
        for x, y in val_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            val_acc = val_model(x, y)

    logger.record_mean("val/acc", val_acc.item())

    model.train()
    return logger.data["val/acc"]
