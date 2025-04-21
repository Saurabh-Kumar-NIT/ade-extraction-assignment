"""
Script to compare different entity extraction methods.

This script evaluates and compares various methods for extracting entities
such as drugs, adverse events, and symptoms from medical text. The methods 
compared include a model-based extractor, a rule-based extractor, and a hybrid 
extractor. The accuracy of each method is computed based on ground truth entities 
from annotations and the results are saved in a JSON file.
"""

import json
import os
import logging
from typing import Dict, List, Any

from src.utils.data_loader import DataLoader
from src.models.entity_extractor import EntityExtractor
from src.models.hybrid_extractor import HybridEntityExtractor
from src.rule_based_extractor import extract_entities as rule_based_extract

# Configure logging to display information and errors in the console
logging.basicConfig(
    level=logging.INFO,  # Log messages of INFO level and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Custom log message format
)
logger = logging.getLogger(__name__)  # Initialize logger

def calculate_accuracy(extracted: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
    """
    Compute the accuracy of entity extraction by comparing the extracted entities 
    with the ground truth annotations.
    
    Args:
        extracted: A dictionary where keys are entity types, and values are lists of extracted entities
        ground_truth: A dictionary where keys are entity types, and values are lists of ground truth entities
        
    Returns:
        A float representing the accuracy between 0 and 1
    """
    total_gt = sum(len(v) for v in ground_truth.values())  # Total number of ground truth entities
    if total_gt == 0:
        return 1.0  # Return perfect accuracy if no ground truth entities are present
    
    matched = 0  # Counter for matching entities
    
    # Compare entities for each type (drugs, adverse events, symptoms)
    for entity_type in ["drugs", "adverse_events", "symptoms"]:
        extracted_set = set(extracted[entity_type])
        gt_set = set(ground_truth[entity_type])
        
        matched += len(extracted_set.intersection(gt_set))  # Count matching entities
    
    return matched / total_gt  # Return the proportion of matched entities

def save_results(doc_id: str, original_text: str, processed_text: str, ground_truth: Dict[str, List[str]], 
                 extracted_entities: Dict[str, Dict[str, List[str]]], accuracy: Dict[str, float]) -> None:
    """
    Save the extraction results (including the accuracy) to a JSON file.
    
    Args:
        doc_id: The document ID for which extraction results are generated
        original_text: The original unprocessed text of the document
        processed_text: The processed version of the document text (if any processing was applied)
        ground_truth: The ground truth entities for comparison
        extracted_entities: The extracted entities using different methods (original, rule-based, hybrid)
        accuracy: A dictionary of accuracy scores for each method
    """
    # Construct the result dictionary
    results = {
        "document_id": doc_id,
        "text": {
            "original": original_text,
            "processed": processed_text
        },
        "ground_truth": ground_truth,
        "extractions": extracted_entities,
        "accuracy": accuracy
    }
    
    output_dir = "data/processed"  # Directory to save the results
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    output_file = os.path.join(output_dir, f"{doc_id}_comparison.json")  # File path to save the results
    
    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")  # Log that results were saved

def main():
    """Main function to execute the extraction comparison for a set of documents."""
    # List of document IDs to compare extraction methods on
    doc_ids = ["LIPITOR.1", "LIPITOR.100"]
    
    # Initialize extractors for original, hybrid, and rule-based methods
    original_extractor = EntityExtractor()
    hybrid_extractor = HybridEntityExtractor()
    
    for doc_id in doc_ids:
        logger.info(f"Processing document: {doc_id}")  # Log the document being processed
        
        # Load the document using the data loader
        loader = DataLoader()
        document = loader.load_document(doc_id)
        
        if not document:
            logger.error(f"Document not found: {doc_id}")  # Log an error if the document is not found
            continue
        
        original_text = document["text"]  # Get the original text
        processed_text = original_text  # In this case, processed text is the same as original
        
        # Extract ground truth entities from the document annotations
        ground_truth = loader.extract_entities_from_annotations(document["annotations"])
        logger.info(f"Ground truth entities: {sum(len(v) for v in ground_truth.values())} entities found")
        
        # Initialize a dictionary to hold extracted entities from each method
        extracted_entities = {
            "original": {},
            "rule_based": {},
            "hybrid": {}
        }
        
        # Extract entities using the original model-based extractor
        try:
            original_extractor.load_model()  # Load the original extraction model
            extracted_entities["original"] = original_extractor.extract_entities(processed_text)
            logger.info(f"Original extractor: {sum(len(v) for v in extracted_entities['original'].values())} entities found")
        except Exception as e:
            logger.error(f"Error with original extractor: {e}")  # Log any errors during extraction
            extracted_entities["original"] = {"drugs": [], "adverse_events": [], "symptoms": []}  # Default to empty lists
        
        # Extract entities using the rule-based extractor
        extracted_entities["rule_based"] = rule_based_extract(original_text)
        logger.info(f"Rule-based extractor: {sum(len(v) for v in extracted_entities['rule_based'].values())} entities found")
        
        # Extract entities using the hybrid extractor (combination of models and rules)
        try:
            extracted_entities["hybrid"] = hybrid_extractor.extract_entities(original_text)
            logger.info(f"Hybrid extractor: {sum(len(v) for v in extracted_entities['hybrid'].values())} entities found")
        except Exception as e:
            logger.error(f"Error with hybrid extractor: {e}")  # Log any errors during extraction
            extracted_entities["hybrid"] = {"drugs": [], "adverse_events": [], "symptoms": []}  # Default to empty lists
        
        # Calculate accuracy for each extraction method
        accuracy = {
            "original": calculate_accuracy(extracted_entities["original"], ground_truth),
            "rule_based": calculate_accuracy(extracted_entities["rule_based"], ground_truth),
            "hybrid": calculate_accuracy(extracted_entities["hybrid"], ground_truth)
        }
        
        logger.info(f"Accuracy - Original: {accuracy['original']:.2f}, Rule-based: {accuracy['rule_based']:.2f}, Hybrid: {accuracy['hybrid']:.2f}")
        
        # Save the comparison results to a file
        save_results(doc_id, original_text, processed_text, ground_truth, extracted_entities, accuracy)

if __name__ == "__main__":
    main()  # Run the script when executed
