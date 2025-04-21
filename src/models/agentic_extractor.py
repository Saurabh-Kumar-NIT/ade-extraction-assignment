import logging
import json
import re
from typing import Dict, List, Any, Optional

from src.models.entity_extractor import EntityExtractor
from src.verification.verifier import ExtractionVerifier
from src.utils.preprocessor import TextPreprocessor
from src.utils.config import MAX_RETRIES

logger = logging.getLogger(__name__)

class AgenticExtractor:
    def __init__(self, max_retries: int = MAX_RETRIES):
        self.max_retries = max_retries
        self.entity_extractor = EntityExtractor() 
        self.verifier = ExtractionVerifier()
        self.preprocessor = TextPreprocessor()

    def extract_with_verification(self, text: str, ground_truth: Optional[Dict[str, set[str]]] = None) -> Dict[str, Any]:
        processed_text = self.preprocessor.process_text(text)
        logger.info("Performing initial entity extraction")

        entities = self._extract_entities_with_prompt(processed_text)

        attempt = 1
        while attempt <= self.max_retries:
            verification_result = self.verifier.verify_extraction(entities, processed_text, ground_truth)

            if verification_result["is_valid"]:
                logger.info(f"Extraction succeeded after {attempt} attempts")
                return self._build_result(text, processed_text, entities, verification_result, attempt, True)

            if attempt < self.max_retries:
                logger.info(f"Verification failed. Retry attempt {attempt+1}/{self.max_retries}")
                feedback = self._generate_feedback(verification_result)
                entities = self._extract_entities_with_prompt(processed_text, feedback)

            attempt += 1

        logger.warning(f"Extraction failed after {self.max_retries} attempts")
        return self._build_result(text, processed_text, entities, verification_result, self.max_retries, False)

    def _extract_entities_with_prompt(self, text: str, feedback: str = "") -> Dict[str, List[str]]:
        if feedback:
            prompt = (
                f"Fix these errors in your extraction:\n{feedback}\n\n"
                f"Original text:\n{text}\n\n"
                "Respond ONLY with valid JSON:\n"
                "{ \"drugs\": [\"...\"], \"adverse_events\": [\"...\"], \"symptoms\": [\"...\"] }"
            )
        else:
            prompt = (
                "Extract the following medical entities from the text in ONLY valid JSON format:\n"
                "{ \"drugs\": [\"...\"], \"adverse_events\": [\"...\"], \"symptoms\": [\"...\"] }\n\n"
                f"Text: {text}\n\n"
                "ONLY return valid JSON. Do NOT explain anything."
            )

        raw_output = self._call_model_with_prompt(prompt)
        return self._parse_model_output(raw_output)

    def _call_model_with_prompt(self, prompt: str) -> str:
        inputs = self.entity_extractor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.entity_extractor.device)

        outputs = self.entity_extractor.model.generate(
            inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            temperature=0.3,
            do_sample=False
        )

        return self.entity_extractor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _parse_model_output(self, raw_output: str) -> Dict[str, List[str]]:
        logger.debug(f"Raw model output: {raw_output}")

        try:
            if raw_output.startswith("{") and raw_output.endswith("}"):
                return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("No valid JSON found in output, using empty structure")
        return {"drugs": [], "adverse_events": [], "symptoms": []}

    def _generate_feedback(self, verification_result: Dict[str, Any]) -> str:
        feedback = []
        if not verification_result["format_check"]["passed"]:
            feedback.append(
                "ERROR: You must return ONLY valid JSON with exactly these fields: "
                "{\"drugs\": [\"...\"], \"adverse_events\": [\"...\"], \"symptoms\": [\"...\"]}"
            )
        if not verification_result["completeness_check"]["passed"]:
            feedback.append("ERROR: Some expected entities were missing from your extraction")
        if not verification_result["semantic_check"]["passed"]:
            feedback.append("ERROR: Some extracted entities don't match the text content")

        return "\n".join(feedback)

    def _build_result(self, original_text: str, processed_text: str, 
                      entities: Dict[str, List[str]], verification_result: Dict[str, Any],
                      attempts: int, success: bool) -> Dict[str, Any]:
        standardized_entities = self.entity_extractor.standardize_entities(entities)
        return {
            "original_text": original_text,
            "processed_text": processed_text,
            "entities": entities,
            "standardized_entities": standardized_entities,
            "verification": verification_result,
            "attempts": attempts,
            "success": success
        }










