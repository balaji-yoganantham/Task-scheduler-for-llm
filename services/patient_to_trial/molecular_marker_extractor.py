"""
Enhanced Molecular Marker Extractor
Extracts and infers molecular marker status from patient records with improved detection
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MolecularMarkerExtractor:
    """Enhanced extractor for molecular markers with inference capabilities"""
    
    def __init__(self):
        # Comprehensive patterns for BRAF mutations
        self.braf_patterns = {
            'positive': [
                r'BRAF\s*V600E\s*(?:positive|mutant|mutation|mutated|\+)',
                r'BRAF\s*V600E',
                r'BRAF\s*V600K\s*(?:positive|mutant|mutation|mutated|\+)',
                r'BRAF\s*V600K',
                r'BRAF\s*(?:positive|mutant|mutation|mutated|\+)',
                r'BRAF\s*mutant',
                r'BRAF\s*mutation',
                r'BRAF\s*V600[EK]',
                r'positive\s*for\s*BRAF',
                r'BRAF\s*test\s*positive',
            ],
            'negative': [
                r'BRAF\s*(?:negative|wild[-\s]?type|wt|wildtype|\-)',
                r'BRAF\s*V600E\s*(?:negative|wild[-\s]?type|wt)',
                r'no\s+BRAF\s*(?:mutation|mutant)',
                r'BRAF\s*test\s*negative',
                r'BRAF\s*not\s*(?:mutated|mutant)',
            ],
            'implied_positive': [
                r'BRAF\s*inhibitor',
                r'BRAF\s*targeted',
                r'treatment\s*for\s*BRAF',
            ]
        }
        
        # Comprehensive patterns for KRAS mutations
        self.kras_patterns = {
            'positive': [
                r'KRAS\s*G12C\s*(?:positive|mutant|mutation|mutated|\+)',
                r'KRAS\s*G12C',
                r'KRAS\s*G12D\s*(?:positive|mutant|mutation|mutated|\+)',
                r'KRAS\s*G12D',
                r'KRAS\s*G12V\s*(?:positive|mutant|mutation|mutated|\+)',
                r'KRAS\s*G12V',
                r'KRAS\s*G13D\s*(?:positive|mutant|mutation|mutated|\+)',
                r'KRAS\s*G13D',
                r'KRAS\s*(?:positive|mutant|mutation|mutated|\+)',
                r'KRAS\s*mutant',
                r'KRAS\s*mutation',
                r'positive\s*for\s*KRAS',
                r'KRAS\s*test\s*positive',
            ],
            'negative': [
                r'KRAS\s*(?:negative|wild[-\s]?type|wt|wildtype|\-)',
                r'KRAS\s*G12C\s*(?:negative|wild[-\s]?type|wt)',
                r'no\s+KRAS\s*(?:mutation|mutant)',
                r'KRAS\s*test\s*negative',
                r'KRAS\s*not\s*(?:mutated|mutant)',
            ],
            'implied_positive': [
                r'KRAS\s*G12[CDV]',
                r'KRAS\s*G13D',
            ]
        }
        
        # Comprehensive patterns for MSI status
        self.msi_patterns = {
            'msi_high': [
                r'MSI[-\s]?H\s*(?:positive|high|\+)',
                r'MSI[-\s]?H',
                r'microsatellite\s+instability\s+high',
                r'MSI\s*high',
                r'dMMR\s*(?:positive|high|\+)',
                r'deficient\s+MMR',
                r'mismatch\s+repair\s+deficient',
            ],
            'msi_low': [
                r'MSI[-\s]?L\s*(?:positive|low|\+)',
                r'MSI[-\s]?L',
                r'microsatellite\s+instability\s+low',
                r'MSI\s*low',
            ],
            'mss': [
                r'MSS\s*(?:positive|stable|\+)',
                r'MSS',
                r'microsatellite\s+stable',
                r'MSI\s*stable',
                r'proficient\s+MMR',
                r'mismatch\s+repair\s+proficient',
            ],
            'negative': [
                r'MSI\s*(?:negative|\-)',
                r'no\s+MSI',
            ]
        }
        
        # PD-L1 patterns
        self.pdl1_patterns = {
            'positive': [
                r'PD[-\s]?L1\s*(?:positive|high|\+|>|≥)',
                r'PD[-\s]?L1\s*expression',
                r'TPS\s*(?:>|≥|high)',
                r'CPS\s*(?:>|≥|high)',
            ],
            'negative': [
                r'PD[-\s]?L1\s*(?:negative|low|\-|<|≤)',
                r'PD[-\s]?L1\s*not\s*expressed',
            ]
        }
    
    def extract_braf_status(self, text: str) -> Optional[str]:
        """Extract BRAF mutation status"""
        text_lower = text.lower()
        
        # Check for positive patterns
        for pattern in self.braf_patterns['positive']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Check if it's V600E specifically
                if 'v600e' in text_lower:
                    return 'BRAF V600E positive'
                elif 'v600k' in text_lower:
                    return 'BRAF V600K positive'
                return 'BRAF mutant'
        
        # Check for negative patterns
        for pattern in self.braf_patterns['negative']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'BRAF wild-type'
        
        # Check for implied positive
        for pattern in self.braf_patterns['implied_positive']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'BRAF mutant (implied)'
        
        return None
    
    def extract_kras_status(self, text: str) -> Optional[str]:
        """Extract KRAS mutation status"""
        text_lower = text.lower()
        
        # Check for specific mutations first
        specific_mutations = {
            'g12c': 'KRAS G12C',
            'g12d': 'KRAS G12D',
            'g12v': 'KRAS G12V',
            'g13d': 'KRAS G13D',
        }
        
        for mutation_key, mutation_name in specific_mutations.items():
            pattern = rf'KRAS\s*{mutation_key}\s*(?:positive|mutant|mutation|mutated|\+)?'
            if re.search(pattern, text_lower, re.IGNORECASE):
                return f'{mutation_name} positive'
        
        # Check for general positive patterns
        for pattern in self.kras_patterns['positive']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'KRAS mutant'
        
        # Check for negative patterns
        for pattern in self.kras_patterns['negative']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'KRAS wild-type'
        
        # Check for implied positive (specific mutations mentioned)
        for pattern in self.kras_patterns['implied_positive']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    mutation = match.group(0).upper()
                    return f'{mutation} positive (implied)'
        
        return None
    
    def extract_msi_status(self, text: str) -> Optional[str]:
        """Extract MSI status"""
        text_lower = text.lower()
        
        # Check for MSI-H
        for pattern in self.msi_patterns['msi_high']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'MSI-H'
        
        # Check for MSI-L
        for pattern in self.msi_patterns['msi_low']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'MSI-L'
        
        # Check for MSS
        for pattern in self.msi_patterns['mss']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'MSS'
        
        # Check for negative
        for pattern in self.msi_patterns['negative']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'MSI negative'
        
        return None
    
    def extract_pdl1_status(self, text: str) -> Optional[str]:
        """Extract PD-L1 status"""
        text_lower = text.lower()
        
        # Check for positive
        for pattern in self.pdl1_patterns['positive']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'PD-L1 positive'
        
        # Check for negative
        for pattern in self.pdl1_patterns['negative']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'PD-L1 negative'
        
        return None
    
    def infer_marker_status(self, text: str, diagnosis: str, context: Dict) -> Dict[str, Optional[str]]:
        """
        Infer molecular marker status from context when not explicitly stated
        Uses clinical reasoning based on diagnosis, stage, and treatment history
        """
        inferred = {}
        text_lower = text.lower()
        diagnosis_lower = diagnosis.lower()
        
        # For colorectal cancer patients
        if 'colorectal' in diagnosis_lower or 'colon' in diagnosis_lower or 'rectal' in diagnosis_lower:
            # If patient has metastatic CRC and no mutation mentioned, could be wild-type
            # But we can't assume - this is context-dependent
            if 'metastatic' in text_lower and 'stage iv' in text_lower:
                # Check if there's any mention of testing
                if 'test' in text_lower or 'biomarker' in text_lower or 'molecular' in text_lower:
                    # Testing was done but results not mentioned - could infer unknown
                    pass
                # If no testing mentioned at all, might infer wild-type for common mutations
                # But this is risky - better to leave as unknown
        
        # If patient is on BRAF inhibitor, likely BRAF mutant
        if 'braf inhibitor' in text_lower or 'encorafenib' in text_lower or 'dabrafenib' in text_lower:
            inferred['BRAF'] = 'BRAF mutant (inferred from treatment)'
        
        # If patient is on KRAS G12C inhibitor, likely KRAS G12C
        if 'sotorasib' in text_lower or 'adagrasib' in text_lower:
            inferred['KRAS'] = 'KRAS G12C positive (inferred from treatment)'
        
        # If patient has Lynch syndrome or family history, might be MSI-H
        if 'lynch' in text_lower or 'hnpcc' in text_lower:
            inferred['MSI'] = 'MSI-H (inferred from Lynch syndrome)'
        
        return inferred
    
    def extract_all_markers(self, text: str, diagnosis: str = "", context: Dict = None) -> Dict[str, Optional[str]]:
        """
        Extract all molecular markers from text with inference
        """
        if context is None:
            context = {}
        
        markers = {}
        
        # Extract explicit markers
        braf = self.extract_braf_status(text)
        if braf:
            markers['BRAF'] = braf
        
        kras = self.extract_kras_status(text)
        if kras:
            markers['KRAS'] = kras
        
        msi = self.extract_msi_status(text)
        if msi:
            markers['MSI'] = msi
        
        pdl1 = self.extract_pdl1_status(text)
        if pdl1:
            markers['PD-L1'] = pdl1
        
        # Infer markers from context
        inferred = self.infer_marker_status(text, diagnosis, context)
        for marker, status in inferred.items():
            if marker not in markers:  # Don't override explicit findings
                markers[marker] = status
        
        return markers

