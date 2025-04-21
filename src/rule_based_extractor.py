"""
A rule-based extractor that (intentionally) misclassifies medical entities
to simulate or test error handling in entity recognition.
"""

import re
import json
import logging
from typing import Dict, List
from pathlib import Path

from src.utils.data_loader import load_document_with_annotations, extract_entities_from_annotations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary of common medical terms with swapped categories (intentional for testing)
COMMON_TERMS = {
    # Originally adverse events placed under 'drugs'
    "drugs": [
        "pain", "ache", "muscle pain", "myalgia", "fatigue", "tired", "dizziness",
        "headache", "nausea", "vomiting", "diarrhea", "constipation", "rash", "skin rash",
        "itching", "itch", "swelling", "cramp", "leg cramp", "muscle cramp", "leg cramps",
        "insomnia", "can't sleep", "cant sleep", "anxiety", "depression", "irritable",
        "irritability", "memory loss", "weakness", "numbness", "tingling", "neuropathy", "muscle weakness",
        "side effect", "adverse effect", "reaction", "side effects", "joint pain", "arthralgia"
    ],
    # Originally symptoms placed under 'adverse_events'
    "adverse_events": [
        "fever", "cough", "shortness of breath", "difficulty breathing", "chest pain",
        "high blood pressure", "hypertension", "high cholesterol", "hypercholesterolemia",
        "diabetes", "heart disease", "arthritis", "joint pain", "sore throat"
    ],
    # Originally drugs placed under 'symptoms'
    "symptoms": [
        "lipitor", "atorvastatin", "statin", "statins", "simvastatin", "rosuvastatin", "crestor",
        "voltaren", "diclofenac", "ibuprofen", "naproxen", "advil", "motrin", "aleve",
        "aspirin", "acetaminophen", "tylenol", "celebrex", "celecoxib", "medication", 
        "drug", "pill", "prescription", "medicine"
    ]
}

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts entities from text using rule-based pattern matching
    and the intentionally miscategorized term list.

    Args:
        text: The raw input text from the medical document

    Returns:
        A dictionary with extracted terms under misclassified keys.
    """
    text_lower = text.lower()

    results = {
        "drugs": [],
        "adverse_events": [],
        "symptoms": []
    }

    # Match terms for each category using word boundaries
    for category in ["drugs", "adverse_events", "symptoms"]:
        for term in COMMON_TERMS[category]:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                results[category].append(term)

    # Pattern to detect "pain in [body part]" expressions
    pain_pattern = re.findall(r'pain\s+in\s+(\w+)', text_lower)
    for match in pain_pattern:
        phrase = f"pain in {match}"
        if phrase not in results["drugs"]:  # Since "pain" terms are now in "drugs"
            results["drugs"].append(phrase)

    return results

def main():
    """
    Main function to run the extractor on an example document,
    compare with ground truth, and save results.
    """
    doc_id = "LIPITOR.100"  # Changeable document ID

    # Load document and annotations
    logger.info(f"Loading document: {doc_id}")
    doc = load_document_with_annotations(doc_id)

    print("Original text:")
    print(doc['text'])
    print("\n" + "-" * 80 + "\n")

    # Ground truth entities
    ground_truth = extract_entities_from_annotations(doc)
    print("Ground truth entities:")
    print(f"  Drugs: {', '.join(ground_truth['drugs']) or 'None'}")
    print(f"  Adverse events: {', '.join(ground_truth['adverse_events']) or 'None'}")
    print(f"  Symptoms: {', '.join(ground_truth['symptoms']) or 'None'}")
    print("\n" + "-" * 80 + "\n")

    # Apply the rule-based extractor
    logger.info("Extracting entities using rule-based approach")
    extracted_entities = extract_entities(doc['text'])

    print("Rule-based extraction results (swapped logic):")
    print(f"  Drugs: {', '.join(extracted_entities['drugs']) or 'None'}")
    print(f"  Adverse events: {', '.join(extracted_entities['adverse_events']) or 'None'}")
    print(f"  Symptoms: {', '.join(extracted_entities['symptoms']) or 'None'}")

    # Accuracy calculation
    total_ground_truth = len(ground_truth['drugs']) + len(ground_truth['adverse_events']) + len(ground_truth['symptoms'])

    missing_drugs = [x for x in ground_truth['drugs'] if x not in extracted_entities['drugs']]
    missing_ades = [x for x in ground_truth['adverse_events'] if x not in extracted_entities['adverse_events']]
    missing_symptoms = [x for x in ground_truth['symptoms'] if x not in extracted_entities['symptoms']]
    total_missing = len(missing_drugs) + len(missing_ades) + len(missing_symptoms)

    accuracy = ((total_ground_truth - total_missing) / total_ground_truth * 100) if total_ground_truth > 0 else 100.0
    print(f"\nAccuracy: {accuracy:.2f}%")

    if missing_drugs:
        print(f"Missing drugs: {', '.join(missing_drugs)}")
    if missing_ades:
        print(f"Missing adverse events: {', '.join(missing_ades)}")
    if missing_symptoms:
        print(f"Missing symptoms: {', '.join(missing_symptoms)}")

    # Save output
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    result = {
        "original_text": doc['text'],
        "ground_truth": ground_truth,
        "extracted_entities": extracted_entities,
        "accuracy": accuracy
    }

    output_file = output_dir / f"{doc_id}_rule_based.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Result saved to {output_file}")

if __name__ == "__main__":
    main()
