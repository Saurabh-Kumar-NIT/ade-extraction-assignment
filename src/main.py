import sys
import os
import json
import logging
import argparse
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add the root directory to PYTHONPATH so imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Use full module paths based on the project structure
from src.utils.data_loader import load_dataset, extract_entities_from_annotations
from src.models.agentic_extractor import AgenticExtractor
from src.utils.config import PROCESSED_DATA_DIR


# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_document(doc: Dict, extractor: AgenticExtractor, use_ground_truth: bool = True) -> Dict[str, Any]:
    """
    Runs entity extraction on a single document.
    
    Args:
        doc: A dictionary containing document data
        extractor: Instance of AgenticExtractor used for processing
        use_ground_truth: Flag to include ground truth comparison
        
    Returns:
        Extraction result as a dictionary
    """
    logger.info(f"Processing document ID: {doc['id']}")
    
    ground_truth = extract_entities_from_annotations(doc) if use_ground_truth else None
    result = extractor.extract_with_verification(doc['text'], ground_truth)
    result['doc_id'] = doc['id']
    
    return result

def save_results(results: List[Dict[str, Any]], output_dir: str = PROCESSED_DATA_DIR) -> None:
    """
    Persists extraction results and summary statistics.
    
    Args:
        results: List of dictionaries with extraction output
        output_dir: Path where files should be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full result set to JSON
    with open(os.path.join(output_dir, 'extraction_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Compile and save summary statistics in CSV
    summary = [
        {
            'doc_id': res['doc_id'],
            'success': res['success'],
            'attempts': res['attempts'],
            'num_drugs': len(res['entities']['drugs']),
            'num_adverse_events': len(res['entities']['adverse_events']),
            'num_symptoms': len(res['entities']['symptoms']),
        }
        for res in results
    ]
    
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'extraction_summary.csv'), index=False)

    # Compute performance metrics
    total = len(results)
    successful = sum(r['success'] for r in results)
    success_rate = successful / total if total else 0
    avg_attempts = sum(r['attempts'] for r in results) / total if total else 0

    stats = {
        'total_documents': total,
        'successful_extractions': successful,
        'success_rate': success_rate,
        'average_attempts': avg_attempts
    }

    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Output saved in: {output_dir}")
    logger.info(f"Extraction success rate: {success_rate:.2%}, Avg. attempts per document: {avg_attempts:.2f}")

def main():
    """Entry point for execution."""
    parser = argparse.ArgumentParser(description="Run the Adverse Drug Event extraction process")
    parser.add_argument('--limit', type=int, help='Restrict processing to N documents')
    parser.add_argument('--no-ground-truth', action='store_true', help='Disable use of ground truth during validation')
    args = parser.parse_args()

    logger.info("Launching ADE extraction workflow...")

    # Instantiate extractor
    extractor = AgenticExtractor()

    # Load data
    logger.info(f"Loading documents (limit={args.limit})")
    documents = load_dataset(limit=args.limit)

    # Process each document
    results = []
    for doc in tqdm(documents, desc="Running extraction"):
        results.append(process_document(doc, extractor, use_ground_truth=not args.no_ground_truth))

    # Save everything
    save_results(results)

    logger.info("All documents processed successfully.")

if __name__ == "__main__":
    main()


