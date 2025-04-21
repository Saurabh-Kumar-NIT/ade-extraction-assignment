"""
Example script to demonstrate the Adverse Drug Event extraction system.
This script processes a single example from the CADEC dataset.
"""
import logging
import json
from pathlib import Path

from src.utils.data_loader import load_document_with_annotations, extract_entities_from_annotations
from src.models.agentic_extractor import AgenticExtractor
from src.utils.preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run an example extraction on a single document."""
    # Choose an example document ID
    doc_id = "LIPITOR.100"  # You can change this to any other document ID
    
    # Load the document
    logger.info(f"Loading document: {doc_id}")
    doc = load_document_with_annotations(doc_id)
    
    print("Original text:")
    print(doc['text'])
    print("\n" + "-" * 80 + "\n")
    
    # Extract ground truth entities
    ground_truth = extract_entities_from_annotations(doc)
    
    print("Ground truth entities:")
    print(f"  Drugs: {', '.join(ground_truth['drugs']) if ground_truth['drugs'] else 'None'}")
    print(f"  Adverse events: {', '.join(ground_truth['adverse_events']) if ground_truth['adverse_events'] else 'None'}")
    print(f"  Symptoms: {', '.join(ground_truth['symptoms']) if ground_truth['symptoms'] else 'None'}")
    print("\n" + "-" * 80 + "\n")
    
    # Preprocess the text
    logger.info("Preprocessing text")
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.process_text(doc['text'])
    
    print("Preprocessed text:")
    print(processed_text)
    print("\n" + "-" * 80 + "\n")
    
    # Initialize the agentic extractor
    logger.info("Initializing agentic extractor")
    extractor = AgenticExtractor()
    
    # Extract entities with verification
    logger.info("Extracting entities")
    result = extractor.extract_with_verification(doc['text'], ground_truth)
    
    # Display the results
    print("Extraction results:")
    print(f"  Success: {result['success']}")
    print(f"  Attempts: {result['attempts']}")
    
    print("\nExtracted entities:")
    print(f"  Drugs: {', '.join(result['entities']['drugs']) if result['entities']['drugs'] else 'None'}")
    print(f"  Adverse events: {', '.join(result['entities']['adverse_events']) if result['entities']['adverse_events'] else 'None'}")
    print(f"  Symptoms: {', '.join(result['entities']['symptoms']) if result['entities']['symptoms'] else 'None'}")
    
    print("\nStandardized entities:")
    print("  Drugs:")
    for drug in result['standardized_entities']['drugs']:
        print(f"    - {drug['original']} → {drug['standardized']} (CUI: {drug['cui']})")
    
    print("  Adverse events:")
    for ade in result['standardized_entities']['adverse_events']:
        print(f"    - {ade['original']} → {ade['standardized']} (CUI: {ade['cui']})")
    
    print("  Symptoms:")
    for symptom in result['standardized_entities']['symptoms']:
        print(f"    - {symptom['original']} → {symptom['standardized']} (CUI: {symptom['cui']})")
    
    # Save the result to a file
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / f"{doc_id}_example.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Result saved to {output_dir / f'{doc_id}_example.json'}")

if __name__ == "__main__":
    main() 