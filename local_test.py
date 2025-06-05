import yaml
import json
from src import Parser

class LocalTest:
    def __init__(self, local_test_config):
        """
        Initializes the LocalTest with the configuration file path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.parser = Parser()
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
        selected_configs = self.local_test_config.get('field_name')

        # Load extraction configuration
        with open(extraction_config_path, 'r') as file:
            extraction_config = json.load(file)
        if selected_configs:
            # Build a set of all field_name values in the loaded config
            valid_fields = {item["field_name"] for item in extraction_config}

            # If you want to warn about any selected_configs that don’t actually exist:
            missing = [sc for sc in selected_configs if sc not in valid_fields]
            if missing:
                print(f"Warning: these selected_configs were not found and will be ignored: {missing}")

            # Now keep only those dicts whose "field_name" is in selected_configs
            extraction_config = [
                item
                for item in extraction_config
                if item.get("field_name") in selected_configs
            ]

        # Initialize Parser and perform document extraction
        self.parser.perform_de(pdf_path, extraction_config, output_json_path)
        print(f"Extraction complete. Data saved to {output_json_path}.")

if __name__ == "__main__":
    local_test_config = "./local_test_params.yml"
    local_test = LocalTest(local_test_config)
    local_test.run()
