"""
Improved text preprocessor that preserves original text for entity extraction.
"""
import sys
import os
import logging
from typing import Dict, Tuple

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class ImprovedTextPreprocessor(TextPreprocessor):
    """
    An improved version of the TextPreprocessor that preserves original text for entity extraction.
    """
    
    def process_text(self, text: str, use_generative_model: bool = False) -> Dict[str, str]:
        """
        Process text for entity extraction, returning both original and normalized versions.
        
        Args:
            text: The text to preprocess
            use_generative_model: Whether to use a generative model for expansion (disabled by default)
            
        Returns:
            Dictionary containing both the original and processed text
        """
        logger.info("Processing text with improved preprocessor")
        
        # Store the original text
        original_text = text
        
        # Basic normalization - lowercase only
        normalized_text = text.lower()
        
        # Light cleaning without removing important information
        normalized_text = self._basic_clean(normalized_text)
        
        # Perform abbreviation expansion if enabled
        if self.expand_abbreviations:
            normalized_text = self._expand_abbreviations(normalized_text, use_generative_model=False)
            # Note: We're explicitly disabling generative model expansion due to issues it caused
        
        return {
            "original": original_text,
            "processed": normalized_text
        }
    
    def _basic_clean(self, text: str) -> str:
        """
        Perform basic cleaning without removing important information.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Only remove excessive whitespace
        text = ' '.join(text.split())
        return text
    
def patch_preprocessor():
    """
    Patch the TextPreprocessor with the improved version.
    """
    # Save the original class for reference
    original_process_text = TextPreprocessor.process_text
    
    # Replace the process_text method with our improved version
    TextPreprocessor.process_text = ImprovedTextPreprocessor.process_text
    TextPreprocessor._basic_clean = ImprovedTextPreprocessor._basic_clean
    
    logger.info("TextPreprocessor has been patched with improved version")
    
    return original_process_text

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply the patch
    original_method = patch_preprocessor()
    
    logger.info("Preprocessor has been improved. Original texts will now be preserved for entity extraction.") 