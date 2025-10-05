#!/usr/bin/env python3
import os 
import sys

def find_unique_extensions(root_dir: str) -> set:
    """
    """
    unique_extensions = set()
    print(f"Searching for extensions in {root_dir}.")

    if not os.path.isdir(root_dir):
        print("Error root directory not found.")
        return unique_extensions
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            _, extension = os.path.splitext(filename)
            if extension:
                unique_extensions.add(extension)
    return unique_extensions


if __name__ == "__main__":
    target_dir = "assets/vllm-0.10.1"
    found_extensions = find_unique_extensions(target_dir)
    if found_extensions:
        print(f"Found {len(found_extensions)} extensions.")
        for ext in sorted(list(found_extensions)):
            print(ext, end=" ")