#!/usr/bin/env python3
"""Analyze package usage in the codebase."""

import ast
import os
from pathlib import Path
from collections import defaultdict
import re

def get_imports_from_file(file_path):
    """Extract all imports from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse the AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                    
        # Also check for dynamic imports in strings (e.g., __import__)
        dynamic_import_pattern = r'__import__\s*\(\s*[\'"]([^\'"\)]+)'
        matches = re.findall(dynamic_import_pattern, content)
        for match in matches:
            imports.add(match.split('.')[0])
            
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return imports

def analyze_codebase():
    """Analyze all Python files in the codebase."""
    project_root = Path("/Users/sunaina/code/tanzania2025/pitch_perfect")
    all_imports = defaultdict(list)
    
    # Get all Python files
    python_files = list(project_root.rglob("*.py"))
    
    # Skip virtual environments and common excluded directories
    excluded_dirs = {'venv', 'env', '.venv', '__pycache__', '.git', 'build', 'dist', '.pytest_cache'}
    
    for file_path in python_files:
        # Skip if in excluded directory
        if any(excluded in file_path.parts for excluded in excluded_dirs):
            continue
            
        imports = get_imports_from_file(file_path)
        for imp in imports:
            all_imports[imp].append(str(file_path))
    
    return all_imports

def get_requirements_packages():
    """Get all packages from requirements.txt."""
    packages = {}
    req_file = "/Users/sunaina/code/tanzania2025/pitch_perfect/requirements.txt"
    
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before any version specifiers)
                package = re.split(r'[<>=!]', line)[0].strip()
                packages[package] = line
                
    return packages

def main():
    # Get all imports from codebase
    imports = analyze_codebase()
    
    # Get all packages from requirements.txt
    requirements = get_requirements_packages()
    
    # Analyze usage
    used_packages = set()
    unused_packages = set()
    
    # Map common import names to package names
    import_to_package = {
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'speechbrain': 'speechbrain',
        'PIL': 'Pillow',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
        'speech_features': 'python_speech_features',
        'parselmouth': 'praat-parselmouth',
        'whisper': 'openai-whisper',
        'uvicorn': 'uvicorn[standard]',
    }
    
    print("=== IMPORTED PACKAGES ===")
    for imp, files in sorted(imports.items()):
        package_name = import_to_package.get(imp, imp)
        if package_name in requirements or imp in requirements:
            used_packages.add(package_name if package_name in requirements else imp)
            print(f"{imp}: {len(files)} files")
            for f in files[:3]:  # Show first 3 files
                print(f"  - {f}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more files")
            print()
    
    print("\n=== POTENTIALLY UNUSED PACKAGES ===")
    for package in requirements:
        if package not in used_packages and package not in import_to_package.values():
            # Check if it might be a dependency or tool
            if any(keyword in package for keyword in ['pytest', 'black', 'flake8', 'isort', 'pre-commit', '-cov']):
                print(f"{package} - Development/Testing tool")
            else:
                print(f"{package} - Not found in imports")
            unused_packages.add(package)
    
    print("\n=== SUMMARY ===")
    print(f"Total packages in requirements.txt: {len(requirements)}")
    print(f"Packages with direct imports: {len(used_packages)}")
    print(f"Potentially unused packages: {len(unused_packages)}")
    
    return imports, requirements, used_packages, unused_packages

if __name__ == "__main__":
    main()