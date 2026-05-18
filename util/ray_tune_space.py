from ray import tune

from config import Config


def tune_search_space(config: Config) -> Config:
    config.lr = tune.loguniform(1e-2, 1e-1)
    config.weight_decay = tune.loguniform(1e-6, 1e-5)
    config.momentum = tune.uniform(0.9, 0.99)

    if config.model.name == "SimCLR":
        config.model.temperature = tune.uniform(0.01, 0.1)
        config.proj.proj_size = tune.choice(
            [
                "512 - 128 - 128",
                "512 - 256 - 128",
                "512 - 256 - 256",
            ]
        )
    elif config.model.name == "MoCo":
        config.model.temperature = tune.uniform(0.01, 0.1)
        config.model.momentum = tune.uniform(0.9, 0.99)
        config.proj.proj_size = tune.choice(
            [
                "512 - 128 - 128",
                "512 - 256 - 128",
                "512 - 256 - 256",
            ]
        )
    elif config.model.name == "SwAV":
        config.model.temperature = tune.uniform(0.1, 0.5)
        config.model.eps = tune.uniform(0.01, 1.0)
        config.proj.proj_size = tune.choice(
            [
                "512 - 512 - 128",
                "512 - 512 - 256",
                "512 - 1024 - 512",
                "512 - 2048 - 512",
            ]
        )
    elif config.model.name == "BYOL":
        config.model.tau = tune.uniform(0.9, 1.0)
        config.proj.proj_size = tune.choice(
            [
                "512 - 1024 - 512",
                "512 - 2048 - 512",
                "512 - 4096 - 512",
            ]
        )
        config.pred.pred_size = tune.choice(
            [
                "proj_out_dim - 256 - proj_out_dim",
                "proj_out_dim - 512 - proj_out_dim",
                "proj_out_dim - 1024 - proj_out_dim",
            ]
        )
    elif config.model.name == "SimSiam":
        config.proj.proj_size = tune.choice(
            [
                "512-1024-1024-256",
                "512-1024-1024-512",
                "512-2048-2048-256",
                "512-2048-2048-512",
                "512-4096-4096-256",
                "512-4096-4096-512",
            ]
        )
        config.pred.pred_size = tune.choice(
            [
                "proj_out_dim - 256 - proj_out_dim",
                "proj_out_dim - 512 - proj_out_dim",
                "proj_out_dim - 1024 - proj_out_dim",
                "proj_out_dim - 2048 - proj_out_dim",
            ]
        )
    elif config.model.name == "BarlowTwins":
        config.model.lambd = tune.uniform(1e-3, 1e-2)
        config.proj.proj_size = tune.choice(
            [
                "512 - 2048 - 2048 - 2048",
                "512 - 4096 - 4096 - 4096",
            ]
        )
    elif config.model.name == "VICReg":
        config.model.eps = tune.uniform(1e-5, 1e-3)
        config.model.lambd = tune.uniform(20.0, 30.0)
        config.model.mu = tune.uniform(20.0, 30.0)
        config.model.nu = tune.uniform(1e-3, 1e-2)
        config.proj.proj_size = tune.choice(
            [
                "512 - 2048 - 2048 - 2048",
                "512 - 4096 - 4096 - 4096",
            ]
        )
    else:
        raise ValueError("Unknown tune parameters!")

    return config
