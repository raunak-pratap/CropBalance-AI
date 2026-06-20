import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from typing import Dict

from disease.disease_model import (
    CropDiseaseClassifier,
    DISEASE_META,
    NUM_CLASSES,
    get_inference_transform,
)


CLASS_MAPPING = {
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___healthy": "tomato_healthy",

    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
}


class DiseasePredictor:
    MODEL_PATH = os.path.join("models", "saved", "disease_model.pt")

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_inference_transform()
        self.class_names = None
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        model = CropDiseaseClassifier(num_classes=NUM_CLASSES, pretrained=False)

        if os.path.exists(self.MODEL_PATH):
            ckpt = torch.load(self.MODEL_PATH, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            self.class_names = ckpt["classes"]

            logger.info(
                f"Disease model loaded <- {self.MODEL_PATH} "
                f"(epoch {ckpt.get('epoch', '?')})"
            )
        else:
            logger.warning(f"No trained disease model at {self.MODEL_PATH}")

        return model.to(self.device)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._run(image)

    def predict_from_path(self, path: str) -> Dict:
        image = Image.open(path).convert("RGB")
        return self._run(image)

    def _run(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze()

        top5_values, top5_indices = torch.topk(probs, k=5)

        top_class = self.class_names[top5_indices[0].item()]
        top_conf = top5_values[0].item()

        if top_conf < 0.35:
            top_class = "unknown"

        mapped_class = CLASS_MAPPING.get(top_class)

        if mapped_class is None:
            mapped_class = (
                top_class.lower()
                .replace("___", "_")
                .replace(" ", "_")
                .replace("-", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )

        print("TOP_CLASS:", top_class)
        print("MAPPED_CLASS:", mapped_class)
        print("META_EXISTS:", mapped_class in DISEASE_META)
        print("RAW:", top_class)
        print("MAPPED:", mapped_class)

        meta = DISEASE_META.get(mapped_class, DISEASE_META["unknown"])

        top5 = [
            {
                "disease": self.class_names[idx.item()],
                "confidence": round(val.item(), 4),
            }
            for val, idx in zip(top5_values, top5_indices)
        ]

        return {
            "disease": top_class,
            "confidence": round(top_conf, 4),
            "severity": meta["severity"],
            "treatment_en": meta["en"],
            "treatment_hi": meta["hi"],
            "top5": top5,
        }