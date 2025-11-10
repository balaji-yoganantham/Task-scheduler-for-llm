"""
Analyze partial results from LLM accuracy evaluation
Extracts results from saved LLM prompts/responses
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def extract_llm_response_from_file(filepath: Path) -> Dict[str, Any]:
    """Extract LLM response from saved prompt file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for response section
        if "=== RESPONSE ===" in content:
            response_section = content.split("=== RESPONSE ===")[1]
            # Try to extract JSON from response
            # Look for JSON object
            json_match = re.search(r'\{.*\}', response_section, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
        
        return None
    except Exception as e:
        return None

def analyze_saved_responses():
    """Analyze saved LLM responses"""
    llm_sent_dir = Path("llm_sent")
    if not llm_sent_dir.exists():
        print("No llm_sent directory found")
        return
    
    # Find all trial evaluation files
    eval_files = list(llm_sent_dir.glob("trial_evaluation_single_*.txt"))
    print(f"Found {len(eval_files)} evaluation files")
    
    results = []
    for filepath in sorted(eval_files):
        response = extract_llm_response_from_file(filepath)
        if response:
            results.append({
                "file": filepath.name,
                "response": response
            })
    
    print(f"Extracted {len(results)} valid responses")
    return results

if __name__ == "__main__":
    results = analyze_saved_responses()
    if results:
        print(f"\nAnalyzed {len(results)} saved responses")


