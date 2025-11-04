from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW

from train.dataloader import DataModule
from train.model import QwenExtractionModel
from train.rl import GRPOAgent
from train.utils import ConfigLoader, create_logger, set_global_seed


class TrainingLoop:
    """
    Controller for supervised and GRPO training modes.
    """

    def __init__(self, config_path: str):
        self.config_loader = ConfigLoader(config_path)
        cfg = self.config_loader.as_dict()

        self.training_cfg: Dict = cfg.get("training", {})
        self.data_cfg: Dict = cfg.get("data", {})
        self.model_cfg: Dict = cfg.get("model", {})
        self.rl_cfg: Dict = cfg.get("rl", {})
        self.loss_cfg: Dict = self.training_cfg.get("loss", {})
        self.logging_cfg: Dict = cfg.get("logging", {})

        self.seed = int(self.training_cfg.get("seed", 42))
        set_global_seed(self.seed)

        self.device = torch.device(self.training_cfg.get("device", "mps"))

        self.logger = create_logger("train", log_dir=self.logging_cfg.get("dir"))
        self.logger.info("Initialising training loop on device %s", self.device)

        self.mode = self.training_cfg.get("mode", "supervised")
        self.epochs = int(self.training_cfg.get("epochs", 1))
        self.gradient_clip = float(self.training_cfg.get("gradient_clip", 1.0))
        self.lr = float(self.training_cfg.get("lr", 5e-5))
        self.weight_decay = float(self.training_cfg.get("weight_decay", 0.0))
        self.checkpoint_dir = Path(self.training_cfg.get("checkpoint_dir", "./checkpoints")).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.data_module = DataModule(self.data_cfg)
        self.model_wrapper = QwenExtractionModel(self.model_cfg)
        self.agent = GRPOAgent(self.rl_cfg, loss_cfg=self.loss_cfg)

        self.optimiser = AdamW(self.model_wrapper.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def run(self) -> None:
        loaders = self.data_module.construct_loaders()

        best_val_loss: Optional[float] = None
        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(loaders.train, train=True)
            self.logger.info("Epoch %s train_loss=%.4f", epoch, train_loss)

            val_loss = None
            if loaders.val is not None:
                val_loss = self._run_epoch(loaders.val, train=False)
                self.logger.info("Epoch %s val_loss=%.4f", epoch, val_loss)

            if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

    def _run_epoch(self, loader, train: bool) -> float:
        aggregate_loss = 0.0
        batches = 0

        self.model_wrapper.train() if train else self.model_wrapper.eval()

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.set_grad_enabled(train):
                outputs = self.model_wrapper.forward(**batch)
                logits = outputs.logits
                labels = batch["labels"]

                if self.mode == "grpo" and train:
                    loss = self.agent.compute_loss(logits, labels)
                else:
                    loss = self.agent.supervised_loss(logits, labels)

                if train:
                    self.optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(), self.gradient_clip)
                    self.optimiser.step()

            aggregate_loss += float(loss.detach().cpu())
            batches += 1

        return aggregate_loss / max(batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: Optional[float]) -> None:
        ckpt_path = self.checkpoint_dir / f"epoch{epoch}_loss{val_loss if val_loss is not None else 'na'}.pt"
        state = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": self.model_wrapper.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "config_path": str(self.config_loader.path),
            "mode": self.mode,
        }
        torch.save(state, ckpt_path)
        self.logger.info("Saved checkpoint: %s", ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL for document extraction.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    loop = TrainingLoop(args.config)
    loop.run()


if __name__ == "__main__":
    main()
