import yaml
from src.visual_classify import Classifier


class LocalTestClassify:
    """
    Lightweight runner to test classification routing locally.
    """

    def __init__(self, config_path: str):
        self.local_test_config = self._load_config(config_path)
        self.classifier = Classifier()

    def _load_config(self, config_path: str):
        """
        Load YAML configuration for local classification test.
        """
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def run(self):
        """
        Execute classification and print the selected de_config.
        """
        pdf_path = self.local_test_config.get("pdf_path")
        classification_config_path = self.local_test_config.get(
            "classification_config_path",
            "./classification_config/sample_classification.json",
        )

        best_file, scores = self.classifier.classify(pdf_path, classification_config_path)
        print("Classification scores:", scores)
        if best_file:
            print(f"Selected de_config: {best_file}")
        else:
            print("No de_config passed the minimum score threshold.")


if __name__ == "__main__":
    local_test = LocalTestClassify("./local_test_classify.yml")
    local_test.run()
