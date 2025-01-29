import yaml
import json
from models.model_manager import ModelManager
from src.parser import Parser

class LocalTest:
    def __init__(self, config_path):
        """
        Initializes the LocalTest with the configuration file path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """
        Loads the configuration from the YAML file.

        Returns:
            dict: Configuration parameters.
        """
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def run(self):
        """
        Executes the document extraction process based on the configuration.
        """
        # Load models based on the configuration
        ModelManager.load_config(self.config_path)
        ModelManager.initialize_models()

        # Extract configuration parameters
        pdf_path = self.config.get('pdf_path')
        output_json_path = self.config.get('output_json_path')
        extraction_config_path = self.config.get('extraction_config_path')

        # Load extraction configuration
        with open(extraction_config_path, 'r') as file:
            extraction_config = json.load(file)

        # Initialize Parser and perform document extraction
        parser = Parser(extraction_config)
        parser.perform_de(pdf_path, output_json_path)
        print(f"Extraction complete. Data saved to {output_json_path}.")

if __name__ == "__main__":
    config_path = "config/settings.yml"
    local_test = LocalTest(config_path)
    local_test.run()
