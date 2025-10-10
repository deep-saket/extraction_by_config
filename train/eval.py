from __future__ import annotations

import argparse
from typing import Dict

import torch

from train.dataloader import DataModule
from train.model import QwenExtractionModel
from train.rl import GRPOAgent
from train.utils import ConfigLoader, create_logger


class EvaluatorApp:
    """
    Evaluate a trained Qwen2.5-VL checkpoint on the test split.
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        self.config_loader = ConfigLoader(config_path)
        cfg = self.config_loader.as_dict()

        self.model_cfg: Dict = cfg.get("model", {})
        self.data_cfg: Dict = cfg.get("data", {})
        self.rl_cfg: Dict = cfg.get("rl", {})
        eval_cfg: Dict = cfg.get("evaluation", {})

        self.device = torch.device(eval_cfg.get("device", "mps"))
        self.logger = create_logger("eval", log_dir=eval_cfg.get("log_dir"))

        self.data_module = DataModule(self.data_cfg)
        self.model_wrapper = QwenExtractionModel(self.model_cfg)
        self.agent = GRPOAgent(self.rl_cfg)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_wrapper.model.load_state_dict(checkpoint["model_state"])
        self.model_wrapper.eval()

        self.logger.info("Loaded checkpoint from %s", checkpoint_path)

    def run(self) -> Dict[str, float]:
        loaders = self.data_module.construct_loaders()
        loader = loaders.test or loaders.val
        if loader is None:
            raise ValueError("No evaluation split available.")

        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model_wrapper.forward(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                loss = self.agent.supervised_loss(logits, labels)
                total_loss += float(loss.detach().cpu())
                batches += 1

        average = total_loss / max(batches, 1)
        metrics = {"supervised_loss": average}
        self.logger.info("Evaluation metrics: %s", metrics)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen extraction checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    app = EvaluatorApp(args.config, args.checkpoint)
    app.run()


if __name__ == "__main__":
    main()

