"""
Extract PDF content to JSON format
Extracts text, tables, and metadata from PDF files
"""

import json
import pdfplumber
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys


def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract content from PDF file and convert to JSON
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary containing extracted content
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    result = {
        "file_name": pdf_file.name,
        "file_path": str(pdf_file.absolute()),
        "extraction_timestamp": datetime.now().isoformat(),
        "metadata": {},
        "pages": [],
        "tables": [],
        "full_text": ""
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract metadata
            if pdf.metadata:
                result["metadata"] = {
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "producer": pdf.metadata.get("Producer", ""),
                    "creation_date": str(pdf.metadata.get("CreationDate", "")),
                    "modification_date": str(pdf.metadata.get("ModDate", ""))
                }
            
            result["total_pages"] = len(pdf.pages)
            
            # Extract content from each page
            all_text = []
            
            for page_num, page in enumerate(pdf.pages, start=1):
                page_data = {
                    "page_number": page_num,
                    "text": "",
                    "tables": []
                }
                
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    page_data["text"] = page_text
                    all_text.append(page_text)
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables, start=1):
                        table_data = {
                            "table_number": table_num,
                            "page_number": page_num,
                            "rows": table,
                            "row_count": len(table),
                            "column_count": len(table[0]) if table else 0
                        }
                        page_data["tables"].append(table_data)
                        result["tables"].append(table_data)
                
                result["pages"].append(page_data)
            
            # Combine all text
            result["full_text"] = "\n\n".join(all_text)
            result["total_characters"] = len(result["full_text"])
            result["total_tables"] = len(result["tables"])
            
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        raise
    
    return result


def save_to_json(data: Dict[str, Any], output_path: str = None) -> str:
    """
    Save extracted data to JSON file
    
    Args:
        data: Dictionary to save
        output_path: Optional output path. If not provided, generates from input file name
        
    Returns:
        Path to saved JSON file
    """
    if output_path is None:
        input_file = Path(data["file_path"])
        output_path = input_file.parent / f"{input_file.stem}_extracted.json"
    
    output_file = Path(output_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(output_file)


def main():
    """Main function to extract PDF to JSON"""
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_to_json.py <pdf_file_path> [output_json_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        print(f"📄 Extracting content from: {pdf_path}")
        result = extract_pdf_content(pdf_path)
        
        output_file = save_to_json(result, output_path)
        
        print(f"✅ Successfully extracted PDF content")
        print(f"   - Pages: {result.get('total_pages', 0)}")
        print(f"   - Characters: {result.get('total_characters', 0)}")
        print(f"   - Tables: {result.get('total_tables', 0)}")
        print(f"📁 Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



