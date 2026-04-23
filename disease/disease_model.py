"""
disease_model.py
------------------------
CNN-based crop disease detection via transfer learning (MobileNetV2).

Why MobileNetV2:
  - Only 3.4M parameters → runs on low-end Android/farm tablets
  - ImageNet pre-training → rich visual features out of the box
  - ~20ms inference on CPU → near-real-time on phone cameras
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from loguru import logger
from typing import Dict

# ── 38 disease classes across your 10 supported crops ──────────
DISEASE_CLASSES = [
    # Tomato (9)
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites",
    "tomato_target_spot", "tomato_yellow_leaf_curl_virus", "tomato_healthy",
    # Potato (3)
    "potato_early_blight", "potato_late_blight", "potato_healthy",
    # Rice (4)
    "rice_brown_spot", "rice_hispa", "rice_leaf_blast", "rice_healthy",
    # Wheat (4)
    "wheat_brown_rust", "wheat_yellow_rust", "wheat_loose_smut", "wheat_healthy",
    # Maize (4)
    "maize_common_rust", "maize_gray_leaf_spot", "maize_northern_blight", "maize_healthy",
    # Cotton (3)
    "cotton_bacterial_blight", "cotton_curl_virus", "cotton_healthy",
    # Onion (3)
    "onion_purple_blotch", "onion_stemphylium_blight", "onion_healthy",
    # Soybean (2)
    "soybean_sudden_death", "soybean_healthy",
    # Sugarcane (3)
    "sugarcane_red_rot", "sugarcane_leaf_scald", "sugarcane_healthy",
    # Barley (2)
    "barley_net_blotch", "barley_healthy",
    "unknown",  # fallback
]
NUM_CLASSES = len(DISEASE_CLASSES)

# Severity + bilingual treatment advice (EN + HI)
DISEASE_META: Dict[str, Dict] = {
    "tomato_bacterial_spot":         {"severity": "high",     "en": "Apply copper-based bactericide; remove infected leaves",            "hi": "तांबा जीवाणुनाशक लगाएं; संक्रमित पत्तियां हटाएं"},
    "tomato_early_blight":           {"severity": "medium",   "en": "Spray Mancozeb or Chlorothalonil; rotate crops",                    "hi": "मैनकोजेब छिड़कें; फसल चक्र अपनाएं"},
    "tomato_late_blight":            {"severity": "critical", "en": "Metalaxyl + Mancozeb URGENTLY; destroy severely infected plants",   "hi": "मेटालेक्सिल + मैनकोजेब तुरंत; गंभीर पौधे नष्ट करें"},
    "tomato_leaf_mold":              {"severity": "medium",   "en": "Improve ventilation; apply copper fungicide",                       "hi": "हवादार बनाएं; तांबा फफूंदनाशक छिड़कें"},
    "tomato_septoria_leaf_spot":     {"severity": "medium",   "en": "Remove lower leaves; apply Chlorothalonil",                         "hi": "निचली पत्तियां हटाएं; क्लोरोथैलोनिल लगाएं"},
    "tomato_spider_mites":           {"severity": "medium",   "en": "Neem oil spray; introduce predatory mites",                         "hi": "नीम तेल छिड़कें; शिकारी माइट्स उपयोग करें"},
    "tomato_target_spot":            {"severity": "medium",   "en": "Azoxystrobin fungicide; avoid overhead irrigation",                 "hi": "अजोक्सीस्ट्रोबिन फफूंदनाशक; ऊपर से सिंचाई न करें"},
    "tomato_yellow_leaf_curl_virus": {"severity": "high",     "en": "Control whitefly with imidacloprid; remove infected plants",        "hi": "इमिडाक्लोप्रिड से सफेद मक्खी नियंत्रित करें"},
    "tomato_healthy":                {"severity": "none",     "en": "Crop is healthy — no action needed",                               "hi": "फसल स्वस्थ है"},
    "potato_early_blight":           {"severity": "medium",   "en": "Mancozeb spray; avoid excess nitrogen",                            "hi": "मैनकोजेब छिड़कें; अधिक नाइट्रोजन से बचें"},
    "potato_late_blight":            {"severity": "critical", "en": "Metalaxyl urgently; do NOT store infected tubers",                  "hi": "मेटालेक्सिल तुरंत; संक्रमित कंद न रखें"},
    "potato_healthy":                {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "rice_brown_spot":               {"severity": "high",     "en": "Propiconazole spray; balanced potassium fertilisation",             "hi": "प्रोपिकोनाज़ोल छिड़कें; पोटाशियम संतुलन बनाए"},
    "rice_hispa":                    {"severity": "medium",   "en": "Clip and destroy leaf tips; apply Carbaryl",                        "hi": "पत्ती की नोक काटें; कार्बेरिल लगाएं"},
    "rice_leaf_blast":               {"severity": "critical", "en": "Tricyclazole spray immediately; avoid excess nitrogen",             "hi": "ट्राइसाइक्लाज़ोल तुरंत छिड़कें"},
    "rice_healthy":                  {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "wheat_brown_rust":              {"severity": "high",     "en": "Propiconazole or Tebuconazole; plant resistant varieties next season","hi": "प्रोपिकोनाज़ोल या टेबुकोनाज़ोल छिड़कें"},
    "wheat_yellow_rust":             {"severity": "high",     "en": "Early Tebuconazole application; monitor weekly",                    "hi": "जल्दी टेबुकोनाज़ोल लगाएं; साप्ताहिक निगरानी"},
    "wheat_loose_smut":              {"severity": "medium",   "en": "Seed treatment with Carboxin-Thiram before sowing",                 "hi": "बुआई से पहले कार्बोक्सिन-थिरम से बीजोपचार"},
    "wheat_healthy":                 {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "maize_common_rust":             {"severity": "medium",   "en": "Mancozeb; plant resistant hybrids next season",                     "hi": "मैनकोजेब छिड़कें; प्रतिरोधी संकर किस्में लगाएं"},
    "maize_gray_leaf_spot":          {"severity": "medium",   "en": "Strobilurin fungicide; crop rotation with legumes",                 "hi": "स्ट्रोबिलुरिन फफूंदनाशक; दलहन के साथ फसल चक्र"},
    "maize_northern_blight":         {"severity": "high",     "en": "Azoxystrobin spray; destroy crop debris after harvest",             "hi": "अजोक्सीस्ट्रोबिन छिड़कें; फसल अवशेष नष्ट करें"},
    "maize_healthy":                 {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "cotton_bacterial_blight":       {"severity": "high",     "en": "Copper oxychloride; use certified disease-free seeds",              "hi": "कॉपर ऑक्सीक्लोराइड; प्रमाणित बीज उपयोग करें"},
    "cotton_curl_virus":             {"severity": "critical", "en": "Control whitefly with imidacloprid; use tolerant Bt cotton",        "hi": "सफेद मक्खी नियंत्रित करें; बीटी कॉटन का उपयोग करें"},
    "cotton_healthy":                {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "onion_purple_blotch":           {"severity": "medium",   "en": "Iprodione or Mancozeb; avoid dense planting",                       "hi": "इप्रोडियोन या मैनकोजेब; घनी बुआई से बचें"},
    "onion_stemphylium_blight":      {"severity": "medium",   "en": "Tebuconazole spray; improve field drainage",                        "hi": "टेबुकोनाज़ोल छिड़कें; जल निकासी सुधारें"},
    "onion_healthy":                 {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "soybean_sudden_death":          {"severity": "high",     "en": "Fluopyram seed treatment; improve soil drainage",                   "hi": "फ्लूओपायरम बीजोपचार; जल निकासी सुधारें"},
    "soybean_healthy":               {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "sugarcane_red_rot":             {"severity": "critical", "en": "Use disease-free setts; Carbendazim treatment; destroy infected plants","hi": "स्वस्थ गन्ना रोपें; कार्बेन्डाजिम उपचार करें"},
    "sugarcane_leaf_scald":          {"severity": "high",     "en": "Hot-water treatment of seed cane; copper spray",                    "hi": "गर्म पानी उपचार; तांबा छिड़कें"},
    "sugarcane_healthy":             {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "barley_net_blotch":             {"severity": "medium",   "en": "Propiconazole spray; plant resistant varieties",                    "hi": "प्रोपिकोनाज़ोल छिड़कें; प्रतिरोधी किस्में उगाएं"},
    "barley_healthy":                {"severity": "none",     "en": "Crop is healthy",                                                   "hi": "फसल स्वस्थ है"},
    "unknown":                       {"severity": "unknown",  "en": "Image unclear — retake in daylight focusing on affected leaf",      "hi": "छवि अस्पष्ट — दिन की रोशनी में पुनः फोटो लें"},
}


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_transform() -> transforms.Compose:
    """Heavy augmentation — simulates real-world farm photography."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CropDiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights  = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Freeze early layers — keep low-level edge/texture features
        for i, param in enumerate(backbone.parameters()):
            if i < 80:
                param.requires_grad = False

        # Disease-specific head (replaces ImageNet 1000-class head)
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        self.backbone = backbone
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"CropDiseaseClassifier | classes={num_classes} | trainable={trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_disease_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> CropDiseaseClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CropDiseaseClassifier(num_classes=num_classes, pretrained=pretrained).to(device)