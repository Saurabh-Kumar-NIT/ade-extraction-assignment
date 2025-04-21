"""
UMLS API integration for entity standardization.
"""
import os
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Tuple

from src.utils.config import UMLS_API_KEY, UMLS_API_URL

logger = logging.getLogger(__name__)

class UMLSStandardizer:
    """UMLS API integration for entity standardization."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the UMLS standardizer.
        
        Args:
            api_key: UMLS API key for authentication
        """
        self.api_key = api_key or UMLS_API_KEY
        self.api_url = UMLS_API_URL
        self.tgt = None  # Ticket Granting Ticket
        self.tgt_timestamp = 0
        self.tgt_expiration = 8 * 60 * 60  # 8 hours in seconds
        
        # Load the mock database for development/demo if no API key is available
        self.mock_db = self._load_mock_db()
    
    def _load_mock_db(self) -> Dict:
        """
        Load a mock database of UMLS concepts for development/demo.
        
        Returns:
            Dict containing mock UMLS data
        """
        # Simple mock database for demonstration purposes
        return {
            "drugs": {
                "lipitor": {"cui": "C0286651", "preferred_name": "atorvastatin"},
                "atorvastatin": {"cui": "C0286651", "preferred_name": "atorvastatin"},
                "simvastatin": {"cui": "C0074554", "preferred_name": "simvastatin"},
                "zocor": {"cui": "C0074554", "preferred_name": "simvastatin"},
                "crestor": {"cui": "C0422583", "preferred_name": "rosuvastatin"},
                "rosuvastatin": {"cui": "C0422583", "preferred_name": "rosuvastatin"},
                "voltaren": {"cui": "C0012091", "preferred_name": "diclofenac"},
                "diclofenac": {"cui": "C0012091", "preferred_name": "diclofenac"},
                "ibuprofen": {"cui": "C0020740", "preferred_name": "ibuprofen"},
                "advil": {"cui": "C0020740", "preferred_name": "ibuprofen"},
                "motrin": {"cui": "C0020740", "preferred_name": "ibuprofen"},
                "naproxen": {"cui": "C0027396", "preferred_name": "naproxen"},
                "aleve": {"cui": "C0027396", "preferred_name": "naproxen"}
            },
            "adverse_events": {
                "muscle pain": {"cui": "C0231528", "preferred_name": "myalgia"},
                "myalgia": {"cui": "C0231528", "preferred_name": "myalgia"},
                "muscle ache": {"cui": "C0231528", "preferred_name": "myalgia"},
                "headache": {"cui": "C0018681", "preferred_name": "headache"},
                "nausea": {"cui": "C0027497", "preferred_name": "nausea"},
                "dizziness": {"cui": "C0012833", "preferred_name": "dizziness"},
                "fatigue": {"cui": "C0015672", "preferred_name": "fatigue"},
                "tired": {"cui": "C0015672", "preferred_name": "fatigue"},
                "weakness": {"cui": "C0004093", "preferred_name": "asthenia"},
                "asthenia": {"cui": "C0004093", "preferred_name": "asthenia"},
                "joint pain": {"cui": "C0022408", "preferred_name": "arthralgia"},
                "arthralgia": {"cui": "C0022408", "preferred_name": "arthralgia"},
                "rash": {"cui": "C0015230", "preferred_name": "rash"},
                "skin rash": {"cui": "C0015230", "preferred_name": "rash"},
                "stomach pain": {"cui": "C0221512", "preferred_name": "abdominal pain"},
                "abdominal pain": {"cui": "C0221512", "preferred_name": "abdominal pain"}
            },
            "symptoms": {
                "fever": {"cui": "C0015967", "preferred_name": "fever"},
                "cough": {"cui": "C0010200", "preferred_name": "cough"},
                "high blood pressure": {"cui": "C0020538", "preferred_name": "hypertension"},
                "hypertension": {"cui": "C0020538", "preferred_name": "hypertension"},
                "high cholesterol": {"cui": "C0848558", "preferred_name": "hypercholesterolemia"},
                "hypercholesterolemia": {"cui": "C0848558", "preferred_name": "hypercholesterolemia"},
                "diabetes": {"cui": "C0011849", "preferred_name": "diabetes mellitus"},
                "diabetes mellitus": {"cui": "C0011849", "preferred_name": "diabetes mellitus"},
                "heart disease": {"cui": "C0018799", "preferred_name": "heart disease"},
                "arthritis": {"cui": "C0003864", "preferred_name": "arthritis"}
            }
        }
    
    def _get_authentication_ticket(self) -> str:
        """
        Get a UMLS authentication ticket.
        
        Returns:
            Authentication ticket string
        """
        current_time = time.time()
        
        # Check if TGT has expired
        if self.tgt and (current_time - self.tgt_timestamp) < self.tgt_expiration:
            return self.tgt
        
        # If API key is missing, warn and return empty string
        if not self.api_key:
            logger.warning("UMLS API key not provided. Using mock data for standardization.")
            return ""
        
        # Get a new ticket
        auth_endpoint = f"{self.api_url}/auth"
        try:
            response = requests.post(auth_endpoint, data={"apikey": self.api_key})
            response.raise_for_status()
            
            self.tgt = response.text
            self.tgt_timestamp = current_time
            logger.info("Successfully obtained new UMLS authentication ticket")
            return self.tgt
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get UMLS authentication ticket: {e}")
            return ""
    
    def _search_umls(self, term: str, search_type: str = "exact") -> Dict:
        """
        Search the UMLS for a term.
        
        Args:
            term: The term to search for
            search_type: The type of search (exact, words, approximate)
            
        Returns:
            UMLS search response
        """
        # If no API key, use mock data
        if not self.api_key:
            if search_type == "drugs":
                entity_type = "drugs"
            elif search_type in ["adverse_events", "ADEs", "adverse_drug_events"]:
                entity_type = "adverse_events"
            else:
                entity_type = "symptoms"
                
            term_lower = term.lower()
            if term_lower in self.mock_db[entity_type]:
                return self.mock_db[entity_type][term_lower]
            
            # Try to find partial matches
            for key, value in self.mock_db[entity_type].items():
                if term_lower in key or key in term_lower:
                    return value
            
            # No match found
            return {"cui": "", "preferred_name": term}
        
        # Get authentication ticket
        ticket = self._get_authentication_ticket()
        if not ticket:
            return {"cui": "", "preferred_name": term}
        
        # Search UMLS
        search_endpoint = f"{self.api_url}/search/current"
        params = {
            "string": term,
            "searchType": search_type,
            "ticket": ticket,
            "returnIdType": "concept",
            "pageSize": 1  # Just get the top result
        }
        
        try:
            response = requests.get(search_endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            if "result" in data and "results" in data["result"] and data["result"]["results"]:
                result = data["result"]["results"][0]
                return {
                    "cui": result.get("ui", ""),
                    "preferred_name": result.get("name", term)
                }
            
            return {"cui": "", "preferred_name": term}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search UMLS for '{term}': {e}")
            return {"cui": "", "preferred_name": term}
    
    def standardize_entity(self, entity: str, entity_type: str) -> Tuple[str, str]:
        """
        Standardize a medical entity using UMLS.
        
        Args:
            entity: The entity text to standardize
            entity_type: The type of entity (drug, ADE, symptom)
            
        Returns:
            Tuple of (standardized entity name, concept ID)
        """
        if not entity.strip():
            return entity, ""
        
        # Map entity type to a more specific vocabulary
        if entity_type.lower() in ["drug", "drugs", "medication", "medications"]:
            # Use RxNorm for drugs
            search_result = self._search_umls(entity, "drugs")
        elif entity_type.lower() in ["ade", "adverse_event", "adverse_events", "side_effect", "side_effects"]:
            # Use SNOMED CT for adverse events
            search_result = self._search_umls(entity, "adverse_events")
        elif entity_type.lower() in ["symptom", "symptoms", "disease", "diseases"]:
            # Use SNOMED CT for symptoms and diseases
            search_result = self._search_umls(entity, "symptoms")
        else:
            # Default search
            search_result = self._search_umls(entity)
        
        return search_result.get("preferred_name", entity), search_result.get("cui", "") 