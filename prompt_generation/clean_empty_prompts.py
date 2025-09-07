#!/usr/bin/env python3
"""
Script to clean empty prompts from generated JSON files
"""

import json
import os
import sys
from pathlib import Path
import shutil

def clean_empty_prompts(json_file, backup=True):
    """Clean empty prompts from a JSON file"""
    try:
        # Create backup if requested
        if backup:
            backup_file = str(json_file) + ".backup"
            shutil.copy2(json_file, backup_file)
            print(f"Created backup: {backup_file}")
        
        # Load the data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data.get('prompts', []))
        cleaned_prompts = []
        
        # Filter out empty prompts
        for prompt_data in data.get('prompts', []):
            prompt_text = prompt_data.get('prompt', '')
            if prompt_text and len(prompt_text.strip()) > 0:
                cleaned_prompts.append(prompt_data)
        
        # Update the data
        data['prompts'] = cleaned_prompts
        data['metadata']['total_count'] = len(cleaned_prompts)
        
        # Add cleaning info to metadata
        if 'cleaning_info' not in data['metadata']:
            data['metadata']['cleaning_info'] = {}
        
        data['metadata']['cleaning_info'] = {
            'original_count': original_count,
            'cleaned_count': len(cleaned_prompts),
            'removed_count': original_count - len(cleaned_prompts),
            'cleaned_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Write back to file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Cleaned {json_file}:")
        print(f"  Original prompts: {original_count}")
        print(f"  Cleaned prompts: {len(cleaned_prompts)}")
        print(f"  Removed prompts: {original_count - len(cleaned_prompts)}")
        
        return True
        
    except Exception as e:
        print(f"Error cleaning {json_file}: {e}")
        return False

def main():
    """Main function"""
    import time
    
    output_dir = "/home/gs285/VAR/my_model/prompt_generation/output"
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    # Find all JSON files in the output directory
    json_files = list(Path(output_dir).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {output_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to clean:")
    for json_file in json_files:
        print(f"  - {json_file}")
    
    # Ask for confirmation
    response = input("\nDo you want to clean empty prompts from these files? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    success_count = 0
    for json_file in json_files:
        if clean_empty_prompts(str(json_file)):
            success_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully cleaned {success_count}/{len(json_files)} files")

if __name__ == "__main__":
    main() 