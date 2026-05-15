"""General utilities for the platform."""

import json
import os
import re
from typing import Dict

def load_file(path: str) -> str:
    """
    Read a file and return its contents as text.
    
    Args:
        path (str): The path to the file.
        
    Returns:
        str: File contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_json(data: dict, path: str) -> None:
    """
    Save dict as formatted JSON.
    
    Args:
        data (dict): The dictionary to save.
        path (str): The file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> dict:
    """
    Load JSON from file and return a dictionary.
    
    Args:
        path (str): The file path.
        
    Returns:
        dict: The loaded dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """
    Remove excessive whitespace and non-printable characters.
    
    Args:
        text (str): Input text.
        
    Returns:
        str: Cleaned text.
    """
    # Remove non-printable characters except standard whitespace
    text = "".join(char for char in text if char.isprintable() or char in ["\n", "\r", "\t"])
    # Replace excessive whitespace with a single space, but keep single newlines intact if desired
    # For strict compliance with "remove excessive whitespace", condensing all to single space or just stripping multiple spaces:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_file_extension(path: str) -> str:
    """
    Get the file extension from a path, converted to lowercase.
    
    Args:
        path (str): The file path.
        
    Returns:
        str: Lowercase file extension (e.g., '.pdf').
    """
    _, ext = os.path.splitext(path)
    return ext.lower()
