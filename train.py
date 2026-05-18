from torch import optim
from tqdm.rich import tqdm

from config import Config
from dataloader import get_dataloader
from logger import Logger
from util.ckpt import save
from util.get_module import get_encoder, get_model
from util.log_to_wandb import log_to_wandb
from util.memory_usage import print_memory_usage
from util.set_seed import set_seed

# from memory_profiler import profile


# @profile
def train(config: Config) -> None:
    logger = Logger(file_path=config.exp_path, mode=config.mode)
    logger.info(f"🚀 {config.exp_name}")

    train_loader = get_dataloader(config, "train")

    encoder = get_encoder(config.encoder)

    model = get_model(config.model.name, config, encoder)
    model = model.to(config.device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.num_epoch * len(train_loader)
    )

    for epoch in range(config.num_epoch):
        model.train()
        for x, _ in tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{config.num_epoch}]",
            position=0,
            leave=False,
        ):
            x = [x_i.to(config.device) for x_i in x]

            loss, corr_diff = model(x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.record_mean("train/epoch_loss", loss.item())
            logger.record_mean("train/corr_diff", corr_diff.item())

        scheduler.step()

        logger.record("train/epoch", epoch)
        logger.record("train/lr", optimizer.param_groups[0]["lr"])
        logger.dump(step=epoch)
        logger.info(print_memory_usage())

        if (
            epoch != 0
            and epoch % config.save_freq == 0
            or epoch == (config.num_epoch - 1)
        ):
            save(config, epoch, model, optimizer)

    log_to_wandb(config, config.mode)


if __name__ == "__main__":
    config = Config.parse_args(require_default_file=True)
    config.exp_path.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    train(config)

    config.save_as_yaml(f"{config.exp_path}/config.yaml", indent=4)
