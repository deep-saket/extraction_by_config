import yaml
import json
from src import Parser

class LocalTest:
    def __init__(self, config_path, local_test_config):
        """
        Initializes the LocalTest with the configuration file path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.parser = Parser(self.config_path)
        self.local_test_config = self._load_config(local_test_config)

    def _load_config(self, config_path):
        """
        Loads the configuration from the YAML file.

        Returns:
            dict: Configuration parameters.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def run(self):
        """
        Executes the document extraction process based on the configuration.
        """

        # Extract configuration parameters
        pdf_path = self.local_test_config.get('pdf_path')
        output_json_path = self.local_test_config.get('output_json_path')
        extraction_config_path = self.local_test_config.get('extraction_config_path')

        # Load extraction configuration
        with open(extraction_config_path, 'r') as file:
            extraction_config = json.load(file)

        # Initialize Parser and perform document extraction
        self.parser.perform_de(pdf_path, extraction_config, output_json_path)
        print(f"Extraction complete. Data saved to {output_json_path}.")

if __name__ == "__main__":
    config_path = "config/settings.yml"
    local_test_config = "./local_test_params.yml"
    local_test = LocalTest(config_path, local_test_config)
    local_test.run()
