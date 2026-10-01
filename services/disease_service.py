from typing import Dict

from disease.disease_predictor import DiseasePredictor


class DiseaseService:
    """
    Service layer for CropBalance disease detection.

    This class wraps the existing DiseasePredictor so that
    the future CropBalance Agent does not need to know
    how the neural network works internally.
    """

    def __init__(self):
        self.predictor = DiseasePredictor()

    def detect_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Detect crop disease from image bytes.
        """
        if not image_bytes:
            raise ValueError("Image data is empty.")

        return self.predictor.predict_from_bytes(image_bytes)

    def detect_from_path(self, image_path: str) -> Dict:
        """
        Detect crop disease from an image file path.
        """
        if not image_path:
            raise ValueError("Image path is empty.")

        return self.predictor.predict_from_path(image_path)