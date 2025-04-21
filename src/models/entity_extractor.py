import json
import logging
from typing import Dict, List, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from dotenv import load_dotenv
from pathlib import Path

from src.utils.config import ENTITY_EXTRACTION_MODEL, BASE_DIR
from src.utils.umls import UMLSStandardizer

load_dotenv()
logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extracts medical entities from text using a generative model."""

    def __init__(self, model_name: str = ENTITY_EXTRACTION_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.umls_standardizer = UMLSStandardizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        """Load the extraction model."""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading entity extraction model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            logger.info("Entity extraction model loaded successfully")

    def extract_entities(self, text: str, document_id: Optional[str] = None) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        self.load_model()

        prompt = (
            "Extract the following entities ONLY in a valid JSON format:\n"
            "Return exactly this structure: "
            "{ \"drugs\": [...], \"adverse_events\": [...], \"symptoms\": [...] }\n\n"
            "Text:\n"
            f"{text}\n\n"
            "ONLY return JSON. Do NOT include any explanation or markdown.\n"
            "Begin your response immediately with the JSON object."
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.warning(f"Model raw output before parsing: {raw_output}")
        print(f"\n🟡 RAW MODEL OUTPUT ===> {raw_output}\n")

        # Save raw output for debugging
        try:
            debug_path = BASE_DIR / "debug_raw_output.txt"
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write("\n" + "="*40 + "\n")
                if document_id:
                    f.write("Document ID: " + document_id + "\n")
                f.write("Prompt: \n" + prompt + "\n")
                f.write("\n--- MODEL OUTPUT ---\n")
                f.write(raw_output + "\n")
        except Exception as e:
            logger.error(f"⚠️ Failed to write debug output: {e}")

        # Try parsing output as JSON
        try:
            start_idx = raw_output.find("{")
            end_idx = raw_output.rfind("}") + 1
            json_str = raw_output[start_idx:end_idx]
            logger.debug(f"Trying to parse JSON string:\n{json_str}")
            extracted_entities = json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            extracted_entities = self._fallback_parsing(raw_output)

        if not isinstance(extracted_entities, dict):
            logger.warning("Fallback extraction did not return a dictionary, defaulting to empty structure")
            extracted_entities = {"drugs": [], "adverse_events": [], "symptoms": []}

        # Validate structure
        for field in ["drugs", "adverse_events", "symptoms"]:
            if field not in extracted_entities or not isinstance(extracted_entities[field], list):
                extracted_entities[field] = []

            extracted_entities[field] = [
                e.strip().lower() for e in extracted_entities[field]
                if isinstance(e, str) and e.strip()
            ]

        return extracted_entities

    def _fallback_parsing(self, text: str) -> Dict[str, List[str]]:
        result = {"drugs": [], "adverse_events": [], "symptoms": []}
        lines = text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if "drug" in line.lower() or "medication" in line.lower():
                current_section = "drugs"
                continue
            elif "adverse" in line.lower() or "side effect" in line.lower():
                current_section = "adverse_events"
                continue
            elif "symptom" in line.lower() or "disease" in line.lower():
                current_section = "symptoms"
                continue

            if current_section and line and not line.startswith("{") and not line.endswith("}"):
                clean_line = line.strip('"\'[](){}- \t,')
                if clean_line:
                    result[current_section].append(clean_line)

        return result

    def standardize_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        standardized = {
            "drugs": [],
            "adverse_events": [],
            "symptoms": []
        }

        for drug in entities.get("drugs", []):
            std_name, cui = self.umls_standardizer.standardize_entity(drug, "drug")
            standardized["drugs"].append({
                "original": drug,
                "standardized": std_name if std_name else drug,
                "cui": cui if cui else "N/A"
            })

        for ade in entities.get("adverse_events", []):
            std_name, cui = self.umls_standardizer.standardize_entity(ade, "adverse_event")
            standardized["adverse_events"].append({
                "original": ade,
                "standardized": std_name if std_name else ade,
                "cui": cui if cui else "N/A"
            })

        for symptom in entities.get("symptoms", []):
            std_name, cui = self.umls_standardizer.standardize_entity(symptom, "symptom")
            standardized["symptoms"].append({
                "original": symptom,
                "standardized": std_name if std_name else symptom,
                "cui": cui if cui else "N/A"
            })

        return standardized









 

 

