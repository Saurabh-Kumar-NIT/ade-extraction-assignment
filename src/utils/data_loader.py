"""
Loading files
- Text content
- Original, MedDRA, and SCT annotations
- Extracted entity labels for evaluation
"""

import os
import logging
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from pathlib import Path

from src.utils.config import TEXT_DIR, ORIGINAL_ANNOTATIONS_DIR, MEDDRA_ANNOTATIONS_DIR, SCT_ANNOTATIONS_DIR

logger = logging.getLogger(__name__)

def load_text_file(file_path: str) -> str:
    """
    Load plain text from a file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Cleaned string content from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def parse_ann_file(file_path: str) -> List[Dict]:
    """
    Parse a .ann annotation file and return structured entity information.

    Args:
        file_path (str): Path to the .ann annotation file.

    Returns:
        List[Dict]: Parsed list of annotation dictionaries with type and position info.
    """
    annotations = []

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return annotations  # Return empty list if file doesn't exist or is empty

    with open(file_path, 'r', encoding='utf-8',errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            annotation_id = parts[0]

            # Handle CADEC original annotation format
            if parts[1].startswith(('ADR', 'Drug', 'Finding', 'Disease', 'Symptom')):
                entity_info = parts[1].split(' ')
                if len(entity_info) >= 3:
                    entity_type = entity_info[0]
                    start = int(entity_info[1])

                    # Handle multiple discontinuous spans like "75;96"
                    if ';' in entity_info[2]:
                        end_candidates = map(int, entity_info[2].split(';'))
                        end = max(end_candidates)  # Simplified handling
                    else:
                        end = int(entity_info[2])

                    text = parts[2]

                    annotations.append({
                        'id': annotation_id,
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'text': text
                    })

            # Handle SCT format: "SCTID | Text |"
            elif '|' in parts[1]:
                concept_id = parts[1].split(' | ')[0].strip()
                text = parts[2]

                annotations.append({
                    'id': annotation_id,
                    'concept_id': concept_id,
                    'text': text
                })

            # Handle MedDRA concept ID only format
            else:
                concept_id = parts[1].split(' ')[0]
                text = parts[2]

                annotations.append({
                    'id': annotation_id,
                    'concept_id': concept_id,
                    'text': text
                })

    return annotations

def load_document_with_annotations(doc_id: str) -> Dict:
    """
    Load a document and its annotations by ID.

    Args:
        doc_id (str): Document ID (e.g., 'LIPITOR.1').

    Returns:
        Dict: Full document with text and all annotation sources.
    """
    text_file = os.path.join(TEXT_DIR, f"{doc_id}.txt")
    original_ann_file = os.path.join(ORIGINAL_ANNOTATIONS_DIR, f"{doc_id}.ann")
    meddra_ann_file = os.path.join(MEDDRA_ANNOTATIONS_DIR, f"{doc_id}.ann")
    sct_ann_file = os.path.join(SCT_ANNOTATIONS_DIR, f"{doc_id}.ann")

    text = load_text_file(text_file) if os.path.exists(text_file) else ""

    return {
        'id': doc_id,
        'text': text,
        'original_annotations': parse_ann_file(original_ann_file),
        'meddra_annotations': parse_ann_file(meddra_ann_file),
        'sct_annotations': parse_ann_file(sct_ann_file)
    }

def get_all_document_ids() -> List[str]:
    """
    Retrieve all document IDs in the dataset based on available .txt files.

    Returns:
        List[str]: List of document ID strings.
    """
    doc_ids = [
        file_name[:-4]
        for file_name in os.listdir(TEXT_DIR)
        if file_name.endswith('.txt')
    ]
    return sorted(doc_ids)

def load_dataset(limit: Optional[int] = None) -> List[Dict]:
    """
    Load the CADEC dataset with optional limiting of document count.

    Args:
        limit (Optional[int]): Number of documents to load (None for all).

    Returns:
        List[Dict]: List of parsed document dictionaries.
    """
    doc_ids = get_all_document_ids()
    if limit:
        doc_ids = doc_ids[:limit]

    dataset = [load_document_with_annotations(doc_id) for doc_id in doc_ids]

    logger.info(f"Loaded {len(dataset)} documents from CADEC dataset")
    return dataset

def extract_entities_from_annotations(doc: Dict) -> Dict[str, Set[str]]:
    """
    Extract unique entities from a document's original annotations.

    Args:
        doc (Dict): Document containing annotations.

    Returns:
        Dict[str, Set[str]]: Entity types mapped to lowercase entity mentions.
    """
    entities = {
        'drugs': set(),
        'adverse_events': set(),
        'symptoms': set()
    }

    for ann in doc['original_annotations']:
        if 'type' in ann:
            if ann['type'] == 'Drug':
                entities['drugs'].add(ann['text'].lower())
            elif ann['type'] == 'ADR':
                entities['adverse_events'].add(ann['text'].lower())
            elif ann['type'] in {'Disease', 'Symptom', 'Finding'}:
                entities['symptoms'].add(ann['text'].lower())

    return entities

