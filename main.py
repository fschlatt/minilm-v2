import torch
from lightning.pytorch.cli import LightningCLI
from minilm_v2 import DataModule, MiniLMV2Module


class LinearSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        verbose: bool = False,
    ) -> None:
        last_epoch = -1
        self.interval = "step"

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


class ConstantSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        verbose: bool = False,
    ) -> None:
        last_epoch = -1
        self.interval = "step"

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


def main():
    """
    generate config using `python main.py fit --print_config > config.yaml`
    additional callbacks at:
    https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks

    Example:
        To obtain a default config:

            python main.py fit \
                --trainer.callbacks=ModelCheckpoint \
                --optimizer AdamW \
                --trainer.logger WandbLogger \
                --lr_scheduler LinearSchedulerWithWarmup \
                --print_config > default.yaml

        To run with the default config:

            python main.py fit \
                --config default.yaml

    """
    LightningCLI(MiniLMV2Module, DataModule)


if __name__ == "__main__":
    main()
