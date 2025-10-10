import argparse
import json
from pathlib import Path

import yaml

from src import AutoConfigGenerator


class LocalAutoConfigRunner:
    def __init__(self, config_path: str):
        self.generator = AutoConfigGenerator()
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        path_obj = Path(config_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Auto-config params file not found: {config_path}")
        with path_obj.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def run(self) -> None:
        pdf_path = self.config.get("pdf_path")
        if not pdf_path:
            raise ValueError("`pdf_path` must be specified in the auto-config params file.")

        output_config_path = self.config.get("output_config_path")
        if not output_config_path:
            raise ValueError("`output_config_path` must be specified in the auto-config params file.")

        max_pages = self.config.get("max_pages")

        items = self.generator.generate(
            pdf_path=pdf_path,
            output_config_path=output_config_path,
            max_pages=max_pages,
        )

        print(
            f"Auto configuration completed with {len(items)} items. "
            f"Configuration saved to {output_config_path}."
        )


def main():
    parser = argparse.ArgumentParser(description="Run the AutoConfigGenerator locally.")
    parser.add_argument(
        "--config",
        default="local_test_auto_conf.yml",
        help="Path to the YAML file containing pdf_path, output_config_path, and optional max_pages.",
    )
    args = parser.parse_args()

    runner = LocalAutoConfigRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
