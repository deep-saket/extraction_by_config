from streamlit import checkbox
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import os
import torch
from common import InferenceVisionComponent

class YOLOCheckBox(InferenceVisionComponent):
    def __init__(self, model_name=None, api_endpoint=None, api_token=None, device='cuda'):
        """
        Initialize YOLOv8 model for checkbox detection.

        Args:
            logger: Logger for logging events.
            model_path: Path to the YOLO model, if not provided it will load from yolo_model_path.
        """
        super(InferenceVisionComponent, self).__init__()
        self.model_path = model_name if model_name else None
        self.checkbox_class = [1]

        # Load the model (either pretrained or from scratch)
        if model_name:
            self.model = self.load_model()
        else:
            self.model = self.initialize_model()

    def load_model(self):
        """
        Load the YOLO model from the given path.

        Returns:
            Loaded YOLO model.
        """
        model = YOLO(self.model_path)
        self.logger.info("YOLO model loaded")
        print("YOLO model loaded")
        return model

    def initialize_model(self):
        """
        Initialize the YOLO model from scratch.

        Returns:
            Initialized YOLO model.
        """
        # Assuming YOLO can initialize from scratch with a base configuration
        model = YOLO()  # You might need to adjust this depending on YOLO initialization behavior
        self.logger.info("YOLO model initialized from scratch")
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        """
        Load a pretrained model from a saved directory.

        Args:
            pretrained_model_path: Path to the directory containing model weights and config.
            logger: Logger instance for logging.

        Returns:
            YOLOv8ForCheckBox instance loaded with pretrained weights and config.
        """
        logger = kwargs.get("logger")
        # Initialize the model with configuration settings
        model = cls(logger=logger,
                    model_path=os.path.join(pretrained_model_path, "model.pt"),
                    from_pretrained=True)

        # Load model weights
        logger.info(f"Model loaded from {os.path.join(pretrained_model_path, 'model.pt')}")

        return model

    def forward(self, image_path, output_path=None):
        """
        Perform inference on the given image and save the output if required.

        Args:
            image_path: Path to the image for inference.
            output_path: Optional path to save the output image with bounding boxes.

        Returns:
            Results: Inference results with bounding boxes and classes.
        """
        colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 128, 128),
            (255, 255, 255),
        ]

        results = self.model([image_path])  # Run batched inference on a list of images
        orig_image = results[0].orig_img  # Get the original image
        bboxes = results[0].boxes  # Get bounding boxes

        for bbox in bboxes:  # Draw bounding boxes
            klass = int(bbox.cls.item())
            conf = bbox.conf
            xyxy = bbox.xyxy

            top = (int(xyxy[0][0]), int(xyxy[0][1]))
            bottom = (int(xyxy[0][2]), int(xyxy[0][3]))
            orig_image = cv2.rectangle(orig_image, top, bottom, colors[klass], 4)

            # Add confidence score behind the bounding box
            label = f"Conf: {conf.item():.2f}"
            text_color = (244, 67, 54)
            orig_image = cv2.putText(orig_image, label, (top[0] - 120, top[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                     text_color, 3)

        # Save the image if output path is provided
        if output_path:
            plt.imsave(output_path, orig_image)

        return results

    def get_checked_boxes(self, image_path, output_path=None):
        """
        Get the detected checkboxes with bounding boxes and confidence.

        Args:
            image_path: Path to the image for inference.
            output_path: Optional path to save the output image with bounding boxes.

        Returns:
            List of dictionaries containing bbox, confidence, class ID, and checked status.
        """
        checkboxes = []
        results = self.forward(image_path, output_path=output_path)

        for bbox in results[0].boxes:
            bbox_dict = {
                "bbox": {
                    "x1": int(bbox.xyxy[0][0]),
                    "y1": int(bbox.xyxy[0][1]),
                    "x2": int(bbox.xyxy[0][2]),
                    "y2": int(bbox.xyxy[0][3])
                },
                "confidence": bbox.conf.item(),
                "class_id": int(bbox.cls.item()),
                "checked": int(bbox.cls.item()) in self.checkbox_class
            }
            checkboxes.append(bbox_dict)

        checkboxes = sorted(checkboxes, key=lambda x: (x['bbox']['x1'], x['bbox']['y1']))
        return checkboxes
    
    def infer(self, image_data: str):
        self.logger.info(f"Running inference on image: {image_data}")
        base, ext = os.path.splitext(image_data)
        output_path = f"{base}_cb{ext}"
        checkboxes = self.get_checked_boxes(image_data, output_path=output_path)
        self.logger.info(f"Detected checkboxes: {checkboxes}")
        return checkboxes