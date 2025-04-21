"""
Preprocessor module for text normalization and abbreviation expansion.
"""
import re
import logging
from typing import Dict, List, Tuple, Set
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.config import ABBREVIATION_EXPANSION_MODEL
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utilities for medical forum posts."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.abbreviation_model = None
        self.abbreviation_tokenizer = None
        self.common_abbreviations = {
            "sc": "subcutaneous",
            "im": "intramuscular",
            "iv": "intravenous",
            "po": "by mouth",
            "qd": "once a day",
            "qid": "four times a day",
            "tid": "three times a day",
            "bid": "twice a day",
            "prn": "as needed",
            "sx": "symptoms",
            "tx": "treatment",
            "dx": "diagnosis",
            "hx": "history",
            "fx": "fracture",
            "pt": "patient",
            "bp": "blood pressure",
            "temp": "temperature",
            "labs": "laboratories",
            "lab": "laboratory",
            "mos": "months",
            "mo": "month",
            "yrs": "years",
            "yr": "year",
            "mins": "minutes",
            "min": "minute",
            "hrs": "hours",
            "hr": "hour",
            "w/o": "without",
            "w/": "with",
            "mg": "milligrams",
            "med": "medication",
            "meds": "medications",
            "rx": "prescription",
            "dr": "doctor"
}

    def load_models(self):
        """Load the abbreviation expansion model."""
        if self.abbreviation_model is None:
            logger.info(f"Loading abbreviation expansion model: {ABBREVIATION_EXPANSION_MODEL}")
            self.abbreviation_tokenizer = AutoTokenizer.from_pretrained(ABBREVIATION_EXPANSION_MODEL)
            self.abbreviation_model = AutoModelForSeq2SeqLM.from_pretrained(ABBREVIATION_EXPANSION_MODEL)
            logger.info("Abbreviation expansion model loaded successfully")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra spaces, correcting punctuation, etc.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def expand_common_abbreviations(self, text: str) -> str:
        """
        Expand common medical abbreviations using a predefined dictionary.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with common abbreviations expanded
        """
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w/]', '', word).lower()
            
            # Check if it's an abbreviation
            if clean_word in self.common_abbreviations:
                expanded = self.common_abbreviations[clean_word]
                # Replace the abbreviation with its expansion while preserving punctuation
                expanded_word = word.replace(clean_word, expanded)
                expanded_words.append(expanded_word)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def expand_complex_abbreviations(self, text: str) -> str:
        """
        Expand complex medical abbreviations using a generative model.
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with complex abbreviations expanded
        """
        self.load_models()
        
        prompt = f"Expand medical abbreviations in this text: {text}"
        input_ids = self.abbreviation_tokenizer(prompt, return_tensors="pt").input_ids
        
        outputs = self.abbreviation_model.generate(
            input_ids, 
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        
        expanded_text = self.abbreviation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt pattern that might be in the response
        if expanded_text.startswith("Expand medical abbreviations in this text:"):
            expanded_text = expanded_text[len("Expand medical abbreviations in this text:"):].strip()
        
        return expanded_text
    
    def process_text(self, text: str, use_generative_model: bool = True) -> str:
        """
        Process the text by normalizing and expanding abbreviations.
        
        Args:
            text: Input text
            use_generative_model: Whether to use the generative model for expansion
            
        Returns:
            Processed text
        """
        # Normalize text
        normalized_text = self.normalize_text(text)
        
        # Expand common abbreviations first
        text_with_common_expanded = self.expand_common_abbreviations(normalized_text)
        
        # If requested, expand complex abbreviations using the generative model
        if use_generative_model:
            final_text = self.expand_complex_abbreviations(text_with_common_expanded)
        else:
            final_text = text_with_common_expanded
        
        return final_text 