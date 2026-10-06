from typing import Dict

from disease.disease_model import DISEASE_META
from disease.disease_predictor import CLASS_MAPPING


class AgricultureAdviceService:
    """
    Deterministic agriculture-advice service.

    Advice is grounded in the disease metadata already used by the
    disease predictor. It does not invent treatment recommendations.
    """

    @staticmethod
    def _normalize_disease(disease: str) -> str:
        if disease in CLASS_MAPPING:
            return CLASS_MAPPING[disease]

        return (
            disease.lower()
            .replace("___", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )

    def get_advice(
        self,
        disease: str | None = None,
        confidence: float | None = None,
    ) -> Dict:
        if not disease:
            return {
                "status": "missing_input",
                "message": "Disease information is required to provide disease-specific advice.",
            }

        disease_key = self._normalize_disease(disease)
        meta = DISEASE_META.get(disease_key)

        if meta is None:
            return {
                "status": "unknown_disease",
                "disease": disease,
                "message": (
                    "I do not have verified disease-specific advice for this "
                    "diagnosis yet. Please consult a local agricultural expert."
                ),
            }

        return {
            "status": "success",
            "disease": disease,
            "disease_key": disease_key,
            "severity": meta["severity"],
            "confidence": confidence,
            "treatment_en": meta["en"],
            "treatment_hi": meta["hi"],
            "source": "CropBalance disease metadata",
            "advice_type": "disease_treatment",
            "disclaimer": (
                "Treatment guidance is based on the current CropBalance "
                "metadata and should be checked against current local "
                "agricultural recommendations and product labels before use."
            ),
        }
