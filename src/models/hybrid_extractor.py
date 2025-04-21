"""
Hybrid approach for entity extraction that combines rule-based and generative methods.
"""
import json
import logging
import re
from typing import Dict, List, Tuple, Set, Optional, Any
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from src.utils.config import ENTITY_EXTRACTION_MODEL
from src.utils.umls import UMLSStandardizer
from src.rule_based_extractor import extract_entities as rule_based_extract
from src.rule_based_extractor import COMMON_TERMS

logger = logging.getLogger(__name__)

class HybridEntityExtractor:
    """
    Extracts medical entities using a hybrid approach that combines
    rule-based pattern matching with generative model extraction.
    """
    
    def __init__(self, model_name: str = ENTITY_EXTRACTION_MODEL):
        """
        Initialize the hybrid entity extractor.
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.umls_standardizer = UMLSStandardizer()
        
    def load_model(self):
        """Load the extraction model."""
        if self.model is None:
            logger.info(f"Loading entity extraction model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info("Entity extraction model loaded successfully")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using a hybrid approach.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of extracted entities
        """
        # First, extract entities using the rule-based approach
        rule_based_results = rule_based_extract(text)
        logger.info(f"Rule-based extraction found {sum(len(v) for v in rule_based_results.values())} entities")
        
        # If we found a good number of entities from rule-based, skip the generative model
        total_entities = sum(len(v) for v in rule_based_results.values())
        if total_entities >= 3:
            logger.info("Using rule-based results since enough entities were found")
            return rule_based_results
        
        # Try the generative model approach as additional augmentation
        logger.info("Rule-based extraction found few entities, augmenting with generative model")
        generative_results = self._generative_extract(text)
        logger.info(f"Generative model found {sum(len(v) for v in generative_results.values())} entities")
        
        # Combine both results
        combined_results = {
            "drugs": list(set(rule_based_results["drugs"] + generative_results["drugs"])),
            "adverse_events": list(set(rule_based_results["adverse_events"] + generative_results["adverse_events"])),
            "symptoms": list(set(rule_based_results["symptoms"] + generative_results["symptoms"]))
        }
        
        logger.info(f"Combined approach found {sum(len(v) for v in combined_results.values())} entities")
        return combined_results
    
    def _generative_extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using a generative model.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            Dictionary of extracted entities
        """
        self.load_model()
        
        # Prepare the prompt with an example to guide the model
        prompt = (
            f"Extract medications (drugs), adverse drug events (side effects), and symptoms/diseases from this medical text.\n\n"
            f"Example Text: 'I've been taking Lipitor for 2 months and have muscle pain, fatigue, and leg cramps.'\n\n"
            f"Example Output: {{'drugs': ['lipitor'], 'adverse_events': ['muscle pain', 'fatigue', 'leg cramps'], 'symptoms': []}}\n\n"
            f"Text: {text}\n\n"
            f"Output:"
        )
        
        # Tokenize and generate
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        outputs = self.model.generate(
            input_ids, 
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Initialize default result
        extracted_entities = {"drugs": [], "adverse_events": [], "symptoms": []}
        
        # Parse the JSON output
        try:
            # Find the start and end of the JSON
            start_idx = raw_output.find("{")
            end_idx = raw_output.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = raw_output[start_idx:end_idx]
                # Fix common JSON formatting issues
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                
                # Try to parse as JSON
                try:
                    extracted_entities = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON: {json_str}")
                    # If parsing fails, use fallback parsing
                    extracted_entities = self._fallback_parsing(raw_output)
            else:
                # Fallback if we can't find valid JSON
                logger.warning(f"Could not find JSON in model output: {raw_output}")
                extracted_entities = self._fallback_parsing(raw_output)
                
        except Exception as e:
            logger.warning(f"Error in generative extraction: {e}")
            extracted_entities = self._fallback_parsing(raw_output)
        
        # Ensure proper structure
        if not isinstance(extracted_entities, dict):
            extracted_entities = {"drugs": [], "adverse_events": [], "symptoms": []}
        
        # Make sure all required fields are present
        for field in ["drugs", "adverse_events", "symptoms"]:
            if field not in extracted_entities or not isinstance(extracted_entities[field], list):
                extracted_entities[field] = []
        
        # Clean and normalize the entity strings
        for field in ["drugs", "adverse_events", "symptoms"]:
            extracted_entities[field] = [e.strip().lower() for e in extracted_entities[field] if e and isinstance(e, str)]
        
        return extracted_entities
    
    def _fallback_parsing(self, text: str) -> Dict[str, List[str]]:
        """
        Fallback parsing when JSON parsing fails.
        
        Args:
            text: The raw model output text
            
        Returns:
            Dictionary containing extracted entities
        """
        result = {"drugs": [], "adverse_events": [], "symptoms": []}
        
        # Look for common terms in the output
        text_lower = text.lower()
        
        # Check for drugs
        for drug in COMMON_TERMS["drugs"]:
            if drug in text_lower:
                # Make sure it's a word boundary match
                if re.search(r'\b' + re.escape(drug) + r'\b', text_lower):
                    result["drugs"].append(drug)
        
        # Check for adverse events
        for event in COMMON_TERMS["adverse_events"]:
            if event in text_lower:
                # Make sure it's a word boundary match
                if re.search(r'\b' + re.escape(event) + r'\b', text_lower):
                    result["adverse_events"].append(event)
        
        # Check for symptoms
        for symptom in COMMON_TERMS["symptoms"]:
            if symptom in text_lower:
                # Make sure it's a word boundary match
                if re.search(r'\b' + re.escape(symptom) + r'\b', text_lower):
                    result["symptoms"].append(symptom)
        
        # Look for specific sections
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if "drug" in line.lower() or "medication" in line.lower():
                current_section = "drugs"
                continue
            elif "adverse" in line.lower() or "side effect" in line.lower():
                current_section = "adverse_events"
                continue
            elif "symptom" in line.lower() or "disease" in line.lower():
                current_section = "symptoms"
                continue
            
            # Extract entities if in a section
            if current_section and line and not line.startswith("{") and not line.endswith("}"):
                # Clean the line - remove list markers, quotes, etc.
                items = re.findall(r'[\'"]?([^,\'"\[\]{}]+)[\'"]?', line)
                for item in items:
                    clean_item = item.strip().lower()
                    if clean_item and clean_item not in result[current_section]:
                        result[current_section].append(clean_item)
        
        return result
    
    def standardize_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
        """
        Standardize extracted entities using UMLS.
        
        Args:
            entities: Dictionary of extracted entities by type
            
        Returns:
            Dictionary containing standardized entities
        """
        standardized = {
            "drugs": [],
            "adverse_events": [],
            "symptoms": []
        }
        
        # Standardize drugs
        for drug in entities["drugs"]:
            std_name, cui = self.umls_standardizer.standardize_entity(drug, "drug")
            standardized["drugs"].append({
                "original": drug,
                "standardized": std_name,
                "cui": cui
            })
            
        # Standardize adverse events
        for ade in entities["adverse_events"]:
            std_name, cui = self.umls_standardizer.standardize_entity(ade, "adverse_event")
            standardized["adverse_events"].append({
                "original": ade,
                "standardized": std_name,
                "cui": cui
            })
            
        # Standardize symptoms
        for symptom in entities["symptoms"]:
            std_name, cui = self.umls_standardizer.standardize_entity(symptom, "symptom")
            standardized["symptoms"].append({
                "original": symptom,
                "standardized": std_name,
                "cui": cui
            })
        
        return standardized 