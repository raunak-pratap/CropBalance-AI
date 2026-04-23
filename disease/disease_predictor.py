"""
disease_predictor.py
-----------------------------
Loads the trained CNN checkpoint and runs inference on a single leaf image.
Used by the FastAPI endpoint /disease/detect.
"""

import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from typing import Dict, List

from config import PATH_CONFIG
from disease.disease_model import (
    CropDiseaseClassifier, DISEASE_CLASSES, DISEASE_META,
    NUM_CLASSES, get_inference_transform,
)


class DiseasePredictor:
    """
    Usage:
        predictor = DiseasePredictor()
        result    = predictor.predict_from_bytes(image_bytes)
        result    = predictor.predict_from_path("leaf.jpg")
    """

    MODEL_PATH = os.path.join("models", "saved", "disease_model.pt")

    def __init__(self):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_inference_transform()
        self.model     = self._load_model()
        self.model.eval()

    def _load_model(self) -> CropDiseaseClassifier:
        model = CropDiseaseClassifier(num_classes=NUM_CLASSES, pretrained=False)
        if os.path.exists(self.MODEL_PATH):
            ckpt = torch.load(self.MODEL_PATH, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Disease model loaded <- {self.MODEL_PATH} (epoch {ckpt.get('epoch', '?')})")
        else:
            logger.warning(
                f"No trained disease model at {self.MODEL_PATH}. "
                "Run: python disease/disease_trainer.py"
            )
        return model.to(self.device)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._run(image)

    def predict_from_path(self, path: str) -> Dict:
        image = Image.open(path).convert("RGB")
        return self._run(image)

    def _run(self, image: Image.Image) -> Dict:
        """
        Returns:
            {
                "disease":      "tomato_late_blight",
                "confidence":   0.94,
                "severity":     "critical",
                "treatment_en": "Metalaxyl + Mancozeb URGENTLY...",
                "treatment_hi": "मेटालेक्सिल + मैनकोजेब तुरंत...",
                "top5":         [ {disease, confidence}, ... ]
            }
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze()

        top5_values, top5_indices = torch.topk(probs, k=5)

        top_class = DISEASE_CLASSES[top5_indices[0].item()]
        top_conf  = top5_values[0].item()

        # If confidence is very low, return 'unknown'
        if top_conf < 0.35:
            top_class = "unknown"

        meta = DISEASE_META.get(top_class, DISEASE_META["unknown"])

        top5 = [
            {
                "disease":    DISEASE_CLASSES[idx.item()],
                "confidence": round(val.item(), 4),
            }
            for val, idx in zip(top5_values, top5_indices)
        ]

        return {
            "disease":      top_class,
            "confidence":   round(top_conf, 4),
            "severity":     meta["severity"],
            "treatment_en": meta["en"],
            "treatment_hi": meta["hi"],
            "top5":         top5,
        }