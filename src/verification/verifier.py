"""
Verification system for entity extraction results.

This module provides a set of verification methods to evaluate the results of 
entity extraction. It includes:
1. **Format Verification**: Checks if the extraction result follows the expected schema.
2. **Completeness Verification**: Ensures all entities from the ground truth are captured.
3. **Semantic Similarity Verification**: Assesses if the extracted entities are contextually relevant to the original text.
"""

import logging
import json
from typing import Dict, List, Tuple, Set, Any, Optional
import jsonschema
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from src.utils.config import SENTENCE_TRANSFORMER_MODEL, EXTRACTED_ENTITY_SCHEMA, SIMILARITY_THRESHOLD

# Initialize the logger for this module
logger = logging.getLogger(__name__)

class ExtractionVerifier:
    """
    Verifies entity extraction results using multiple criteria:
    - Format verification (checks if the result matches the required schema)
    - Completeness verification (ensures all ground truth entities are extracted)
    - Semantic similarity verification (checks if entities are contextually related to the text)
    """
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize the extraction verifier.
        
        Args:
            similarity_threshold: A float representing the threshold for semantic similarity 
                                   verification between the extracted entities and the text.
        """
        self.similarity_threshold = similarity_threshold
        self.sentence_model = None  # Will be initialized when needed
    
    def load_sentence_model(self):
        """Load the sentence transformer model for semantic similarity checks."""
        if self.sentence_model is None:
            logger.info(f"Loading sentence transformer model: {SENTENCE_TRANSFORMER_MODEL}")
            self.sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            logger.info("Sentence transformer model loaded successfully")
    
    def verify_format(self, extraction_result: Dict) -> Tuple[bool, str]:
        """
        Verify that the extraction result adheres to the expected schema format.
        
        Args:
            extraction_result: The result of entity extraction to validate.
            
        Returns:
            A tuple (is_valid, error_message) where:
                - is_valid: True if the result follows the correct format, False otherwise.
                - error_message: A string describing the validation error (if any).
        """
        try:
            # Validate the extraction result using the predefined schema
            jsonschema.validate(instance=extraction_result, schema=EXTRACTED_ENTITY_SCHEMA)
            return True, ""  # Validation passed
        except jsonschema.exceptions.ValidationError as e:
            error_message = f"Format verification failed: {e.message}"
            return False, error_message  # Validation failed
    
    def verify_completeness(self, 
                          extraction_result: Dict[str, List[str]], 
                          ground_truth: Dict[str, Set[str]]) -> Tuple[bool, str]:
        """
        Verify that the extracted entities cover all entities present in the ground truth.
        
        Args:
            extraction_result: The extraction result to check for completeness.
            ground_truth: The set of ground truth entities that should be present in the result.
            
        Returns:
            A tuple (is_valid, error_message) where:
                - is_valid: True if all ground truth entities are present, False otherwise.
                - error_message: A string listing missing entities (if any).
        """
        missing_entities = {
            "drugs": set(),
            "adverse_events": set(),
            "symptoms": set()
        }
        
        # Check if each ground truth entity is missing from the extraction result
        for drug in ground_truth["drugs"]:
            if drug not in extraction_result["drugs"]:
                missing_entities["drugs"].add(drug)
        
        for ade in ground_truth["adverse_events"]:
            if ade not in extraction_result["adverse_events"]:
                missing_entities["adverse_events"].add(ade)
        
        for symptom in ground_truth["symptoms"]:
            if symptom not in extraction_result["symptoms"]:
                missing_entities["symptoms"].add(symptom)
        
        # Check if any entities are missing
        is_valid = all(len(missing) == 0 for missing in missing_entities.values())
        
        # Construct error message if there are missing entities
        error_message = ""
        if not is_valid:
            missing_items = []
            if missing_entities["drugs"]:
                missing_items.append(f"Drugs: {', '.join(missing_entities['drugs'])}")
            if missing_entities["adverse_events"]:
                missing_items.append(f"Adverse events: {', '.join(missing_entities['adverse_events'])}")
            if missing_entities["symptoms"]:
                missing_items.append(f"Symptoms: {', '.join(missing_entities['symptoms'])}")
            
            error_message = f"Completeness verification failed. Missing entities: {'; '.join(missing_items)}"
        
        return is_valid, error_message
    
    def verify_semantic_similarity(self, 
                                 extraction_result: Dict[str, List[str]], 
                                 text: str) -> Tuple[bool, str]:
        """
        Verify that the extracted entities are semantically relevant to the original text.
        
        Args:
            extraction_result: The extraction result to check for semantic similarity.
            text: The original text to which the extracted entities should be related.
            
        Returns:
            A tuple (is_valid, error_message) where:
                - is_valid: True if all extracted entities are contextually relevant, False otherwise.
                - error_message: A string describing entities with low semantic similarity (if any).
        """
        self.load_sentence_model()
        
        # Generate embedding for the original text
        text_embedding = self.sentence_model.encode(text)
        
        # Combine all extracted entities (drugs, adverse events, symptoms) into one list
        all_entities = (
            extraction_result["drugs"] + 
            extraction_result["adverse_events"] + 
            extraction_result["symptoms"]
        )
        
        if not all_entities:
            return True, ""  # If no entities are found, no need to check semantic similarity
        
        # Generate embeddings for each entity
        entity_embeddings = self.sentence_model.encode(all_entities)
        
        # Calculate cosine similarity between the text and each entity
        similarities = [np.dot(text_embedding, entity_embedding) / 
                       (np.linalg.norm(text_embedding) * np.linalg.norm(entity_embedding))
                       for entity_embedding in entity_embeddings]
        
        # Check if any entity has similarity below the threshold
        low_similarity_entities = [
            (entity, similarity) 
            for entity, similarity in zip(all_entities, similarities) 
            if similarity < self.similarity_threshold
        ]
        
        # If no entity has low similarity, verification passed
        is_valid = len(low_similarity_entities) == 0
        
        # Construct error message if some entities have low similarity
        error_message = ""
        if not is_valid:
            entity_details = "; ".join([f"'{entity}' (similarity: {similarity:.2f})" 
                                        for entity, similarity in low_similarity_entities])
            error_message = (
                f"Semantic similarity verification failed. "
                f"The following entities have low similarity to the text: {entity_details}"
            )
        
        return is_valid, error_message
    
    def verify_extraction(self, 
                         extraction_result: Dict[str, List[str]], 
                         original_text: str,
                         ground_truth: Optional[Dict[str, Set[str]]] = None) -> Dict[str, Any]:
        """
        Run all verification checks (format, completeness, and semantic similarity) on the extraction result.
        
        Args:
            extraction_result: The result of entity extraction to be verified.
            original_text: The original text to verify the extracted entities against.
            ground_truth: Optional, the ground truth entities to check for completeness (if provided).
            
        Returns:
            A dictionary containing the results of each verification check.
        """
        verification_result = {
            "is_valid": True,  # Final validity flag
            "format_check": {
                "passed": True,
                "error": ""
            },
            "completeness_check": {
                "passed": True,
                "error": ""
            },
            "semantic_check": {
                "passed": True,
                "error": ""
            }
        }
        
        # Perform format verification
        format_valid, format_error = self.verify_format(extraction_result)
        verification_result["format_check"]["passed"] = format_valid
        verification_result["format_check"]["error"] = format_error
        
        # If format verification fails, no need to proceed with further checks
        if not format_valid:
            verification_result["is_valid"] = False
            return verification_result
        
        # Perform completeness verification (only if ground truth is provided)
        if ground_truth:
            completeness_valid, completeness_error = self.verify_completeness(extraction_result, ground_truth)
            verification_result["completeness_check"]["passed"] = completeness_valid
            verification_result["completeness_check"]["error"] = completeness_error
            
            if not completeness_valid:
                verification_result["is_valid"] = False
        
        # Perform semantic similarity verification
        semantic_valid, semantic_error = self.verify_semantic_similarity(extraction_result, original_text)
        verification_result["semantic_check"]["passed"] = semantic_valid
        verification_result["semantic_check"]["error"] = semantic_error
        
        if not semantic_valid:
            verification_result["is_valid"] = False
        
        return verification_result
