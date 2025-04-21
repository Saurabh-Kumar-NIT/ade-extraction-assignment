"""
Enhanced entity extraction pipeline for the CADEC dataset.

Key improvements:
1. Preprocessor now retains the original text structure.
2. Hybrid entity extractor combines rule-based and generative models.
3. Integrated verification step for validating extraction quality.
"""

import json
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Set, Union

from src.utils.data_loader import load_document_with_annotations, extract_entities_from_annotations
from src.models.hybrid_extractor import HybridEntityExtractor
from src.fixes.improved_preprocessor import patch_preprocessor
from src.verification.verifier import ExtractionVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_to_list_converter(obj: Any) -> Any:
    """
    Recursively convert any sets in the object to lists to ensure JSON serialization compatibility.

    Args:
        obj: Input object (could be nested with sets)

    Returns:
        JSON-serializable object with all sets converted to lists
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: set_to_list_converter(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [set_to_list_converter(item) for item in obj]
    return obj

class ImprovedPipeline:
    """
    Improved entity extraction pipeline for processing CADEC documents.
    Applies preprocessing, hybrid extraction, standardization, and verification.
    """
    
    def __init__(self):
        """Initialize components of the pipeline."""
        # Patch preprocessor to retain original text and token mapping
        self.original_preprocessor_method = patch_preprocessor()
        logger.info("Preprocessor patched to preserve original text")
        
        # Load hybrid entity extractor (rule-based + generative)
        self.extractor = HybridEntityExtractor()
        logger.info("Hybrid extractor initialized")
        
        # Load verifier to validate extraction results
        self.verifier = ExtractionVerifier()
        logger.info("Extraction verifier initialized")
    
    def process_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Process a single document through the full pipeline.

        Args:
            doc_id: ID of the document to process

        Returns:
            Dictionary containing original text, extracted entities, standardization, and verification results
        """
        logger.info(f"Processing document: {doc_id}")
        
        # Load document and annotations from disk
        document = load_document_with_annotations(doc_id)
        
        if not document or 'text' not in document or not document['text']:
            logger.error(f"Document not found or empty: {doc_id}")
            return {"error": f"Document not found or empty: {doc_id}"}
        
        # Extract raw input text
        original_text = document["text"]
        
        # Extract ground truth entities from original annotations (if available)
        ground_truth = {}
        if 'original_annotations' in document and document['original_annotations']:
            ground_truth = extract_entities_from_annotations(document)
            logger.info(f"Ground truth entities: {sum(len(v) for v in ground_truth.values())} entities found")
            ground_truth = {k: list(v) for k, v in ground_truth.items()}  # Convert for JSON
        
        # Run hybrid extractor on the input text
        try:
            extracted_entities = self.extractor.extract_entities(original_text)
            logger.info(f"Hybrid extractor found {sum(len(v) for v in extracted_entities.values())} entities")
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            extracted_entities = {"drugs": [], "adverse_events": [], "symptoms": []}
        
        # Standardize extracted entities using UMLS
        try:
            standardized_entities = self.extractor.standardize_entities(extracted_entities)
            logger.info("Entities standardized using UMLS")
        except Exception as e:
            logger.error(f"Error in entity standardization: {e}")
            standardized_entities = {
                "drugs": [],
                "adverse_events": [],
                "symptoms": []
            }
        
        # Convert ground truth back to sets for comparison (if present)
        ground_truth_sets = {k: set(v) for k, v in ground_truth.items()} if ground_truth else None
        
        # Perform verification (compare extracted vs ground truth entities)
        verification_result = self.verifier.verify_extraction(
            extraction_result=extracted_entities,
            original_text=original_text,
            ground_truth=ground_truth_sets
        )
        logger.info(f"Verification complete: {verification_result['is_valid']}")
        
        # Final result structure
        result = {
            "document_id": doc_id,
            "text": {
                "original": original_text
            },
            "extracted_entities": extracted_entities,
            "standardized_entities": standardized_entities,
            "verification": verification_result
        }
        
        if ground_truth:
            result["ground_truth"] = ground_truth
        
        return result
    
    def process_multiple_documents(self, doc_ids: List[str], output_dir: str = "data/processed") -> List[Dict[str, Any]]:
        """
        Process multiple documents and save each result as a JSON file.

        Args:
            doc_ids: List of document IDs
            output_dir: Output folder to save JSON results

        Returns:
            List of result dictionaries for all processed documents
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for doc_id in doc_ids:
            result = self.process_document(doc_id)
            result = set_to_list_converter(result)  # Ensure JSON-safe formatting
            results.append(result)
            
            output_file = os.path.join(output_dir, f"{doc_id}_improved.json")
            try:
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Result saved to {output_file}")
            except TypeError as e:
                logger.error(f"Error saving results: {e}")
        
        return results

def main():
    """
    Run the pipeline on a set of documents specified by command-line arguments.
    Defaults to a predefined sample if no input is provided.
    """
    if len(sys.argv) > 1:
        doc_ids = sys.argv[1:]
    else:
        doc_ids = ["LIPITOR.1", "LIPITOR.100"]
    
    pipeline = ImprovedPipeline()
    pipeline.process_multiple_documents(doc_ids)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()
