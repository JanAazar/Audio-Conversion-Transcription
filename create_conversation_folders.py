import os
from pathlib import Path
import re

def get_highest_conversation_number(folder_path):
    """Find the highest conversation number in the given folder."""
    highest = 0
    if not folder_path.exists():
        return 0
    
    # Pattern to match Conversation_X folders
    pattern = re.compile(r'^Conversation_(\d+)$')
    
    for item in folder_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                num = int(match.group(1))
                highest = max(highest, num)
    
    return highest

def create_conversation_folders():
    """Interactive script to create new conversation folders."""
    recording_dir = Path("Recording")
    
    if not recording_dir.exists():
        print(f"Error: {recording_dir} directory not found!")
        return
    
    # Get list of available subfolders
    available_folders = [d.name for d in recording_dir.iterdir() if d.is_dir()]
    
    if not available_folders:
        print("Error: No subfolders found in Recording directory!")
        return
    
    print("Available subfolders in Recording:")
    for i, folder in enumerate(available_folders, 1):
        print(f"  {i}. {folder}")
    print()
    
    # Ask for subfolder number
    while True:
        try:
            choice = input(f"Enter the number (1-{len(available_folders)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(available_folders):
                subfolder_name = available_folders[choice_num - 1]
                subfolder_path = recording_dir / subfolder_name
                break
            else:
                print(f"Error: Please enter a number between 1 and {len(available_folders)}.")
        except ValueError:
            print("Error: Please enter a valid number.")
    
    # Find the highest conversation number
    highest_num = get_highest_conversation_number(subfolder_path)
    print(f"\nCurrent highest conversation number: {highest_num}")
    
    # Ask for number of folders to create
    while True:
        try:
            num_folders = input(f"\nHow many new conversation folders to create? ").strip()
            num_folders = int(num_folders)
            
            if num_folders <= 0:
                print("Error: Please enter a positive number.")
                continue
            
            break
        except ValueError:
            print("Error: Please enter a valid number.")
    
    # Create the folders
    print(f"\nCreating {num_folders} new conversation folder(s)...")
    created_folders = []
    
    for i in range(1, num_folders + 1):
        new_num = highest_num + i
        folder_name = f"Conversation_{new_num}"
        new_folder_path = subfolder_path / folder_name
        
        if new_folder_path.exists():
            print(f"⚠ Warning: {folder_name} already exists, skipping...")
        else:
            new_folder_path.mkdir(parents=True, exist_ok=True)
            created_folders.append(folder_name)
            print(f"✓ Created: {folder_name}")
    
    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  Subfolder: {subfolder_name}")
    print(f"  Folders created: {len(created_folders)}")
    if created_folders:
        print(f"  Created folders: {', '.join(created_folders)}")
    print("="*60)

if __name__ == "__main__":
    create_conversation_folders()

