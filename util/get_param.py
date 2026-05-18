from config import Config
from prettytable import PrettyTable

from util.get_module import get_encoder, get_model
from util.set_seed import set_seed


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    config = Config.parse_args(require_default_file=True)
    config.exp_path.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    encoder = get_encoder(config.encoder)
    model = get_model(config.model.name, config, encoder)

    count_parameters(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    pytorch_total_params_requires_grad = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"pytorch_total_params = {pytorch_total_params}")
    print(f"pytorch_total_params_requires_grad = {pytorch_total_params_requires_grad}")
