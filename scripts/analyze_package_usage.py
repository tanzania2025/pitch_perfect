#!/usr/bin/env python3
"""Analyze package usage in the codebase."""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple

# Packages to check from requirements.txt
PACKAGES_TO_CHECK = {
    'textblob': 'textblob',
    'python_speech_features': 'python_speech_features',
    'webrtcvad': 'webrtcvad',
    'tiktoken': 'tiktoken',
    'jsonschema': 'jsonschema',
    'python-multipart': 'multipart',
    'pytest-asyncio': 'pytest_asyncio',
    'pytest-mock': 'pytest_mock',
    'torchaudio': 'torchaudio',
    'scipy': 'scipy',
    'scikit-learn': 'sklearn',
    'praat-parselmouth': 'parselmouth',
    'nltk': 'nltk',
    'spacy': 'spacy',
    'vaderSentiment': 'vaderSentiment',
    'pydub': 'pydub',
    'requests': 'requests',
    'aiofiles': 'aiofiles',
    'python-dotenv': 'dotenv',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'transformers': 'transformers',
    'datasets': 'datasets',
    'openai-whisper': 'whisper',
    'librosa': 'librosa',
    'soundfile': 'soundfile',
    'openai': 'openai',
    'pydantic': 'pydantic',
    'elevenlabs': 'elevenlabs',
    'pyyaml': 'yaml',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'pytest': 'pytest',
    'pytest-cov': 'pytest_cov',
    'black': 'black',
    'flake8': 'flake8',
    'isort': 'isort',
    'pre-commit': 'pre_commit',
}

# Development packages
DEV_PACKAGES = {'black', 'flake8', 'isort', 'pre-commit', 'pytest', 'pytest-cov', 'pytest-asyncio', 'pytest-mock'}


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract imports from Python files."""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = defaultdict(set)
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            self.imports.add(base_module)
            for alias in node.names:
                self.from_imports[base_module].add(alias.name)
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Analyze a single Python file for imports."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for dynamic imports
        dynamic_imports = set()
        # Look for importlib usage
        if 'importlib' in content:
            # Simple regex to find potential dynamic imports
            import_patterns = [
                r'importlib\.import_module\(["\']([^"\']+)["\']\)',
                r'__import__\(["\']([^"\']+)["\']\)',
            ]
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                dynamic_imports.update(m.split('.')[0] for m in matches)
        
        # Parse AST
        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        # Combine regular and dynamic imports
        all_imports = analyzer.imports | dynamic_imports
        
        return all_imports, analyzer.from_imports
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return set(), {}


def analyze_codebase(root_dir: Path, exclude_dirs: Set[str] = None) -> Dict[str, List[Path]]:
    """Analyze all Python files in the codebase."""
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.git', '.venv', 'venv', 'env', '.pytest_cache'}
    
    package_usage = defaultdict(list)
    
    for py_file in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
            
        imports, from_imports = analyze_file(py_file)
        
        for package_name, import_name in PACKAGES_TO_CHECK.items():
            if import_name in imports:
                package_usage[package_name].append(py_file)
    
    return package_usage


def check_config_files(root_dir: Path) -> Dict[str, List[Path]]:
    """Check configuration files for package references."""
    config_usage = defaultdict(list)
    
    # Check common config files
    config_patterns = ['*.yaml', '*.yml', '*.json', '*.toml', '*.ini', '*.cfg']
    
    for pattern in config_patterns:
        for config_file in root_dir.rglob(pattern):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for package_name in PACKAGES_TO_CHECK:
                    if package_name.lower() in content:
                        config_usage[package_name].append(config_file)
            except Exception:
                pass
    
    return config_usage


def main():
    """Main analysis function."""
    root_dir = Path(__file__).parent
    
    print("Analyzing package usage in the codebase...\n")
    
    # Analyze Python imports
    package_usage = analyze_codebase(root_dir)
    
    # Check config files
    config_usage = check_config_files(root_dir)
    
    # Categorize packages
    used_packages = set()
    unused_packages = set()
    dev_used_in_main = set()
    
    for package_name in PACKAGES_TO_CHECK:
        files = package_usage.get(package_name, [])
        config_files = config_usage.get(package_name, [])
        
        if files or config_files:
            used_packages.add(package_name)
            
            # Check if dev package is used in non-test code
            if package_name in DEV_PACKAGES:
                non_test_files = [f for f in files if 'test' not in str(f).lower()]
                if non_test_files:
                    dev_used_in_main.add(package_name)
        else:
            unused_packages.add(package_name)
    
    # Generate report
    print("=" * 80)
    print("PACKAGE USAGE ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n## UNUSED PACKAGES (can be removed):")
    print("-" * 40)
    for pkg in sorted(unused_packages):
        print(f"- {pkg}")
    
    print("\n## USED PACKAGES:")
    print("-" * 40)
    for pkg in sorted(used_packages):
        files = package_usage.get(pkg, [])
        config_files = config_usage.get(pkg, [])
        print(f"\n- {pkg}:")
        if files:
            print(f"  Used in {len(files)} Python file(s):")
            for f in files[:3]:  # Show first 3 files
                print(f"    - {f.relative_to(root_dir)}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        if config_files:
            print(f"  Referenced in config files:")
            for f in config_files:
                print(f"    - {f.relative_to(root_dir)}")
    
    print("\n## DEVELOPMENT PACKAGES USED IN MAIN CODE:")
    print("-" * 40)
    if dev_used_in_main:
        for pkg in sorted(dev_used_in_main):
            files = [f for f in package_usage[pkg] if 'test' not in str(f).lower()]
            print(f"- {pkg} used in non-test files:")
            for f in files[:3]:
                print(f"    - {f.relative_to(root_dir)}")
    else:
        print("None (all dev packages appropriately used only in tests)")
    
    print("\n## SUMMARY:")
    print("-" * 40)
    print(f"Total packages checked: {len(PACKAGES_TO_CHECK)}")
    print(f"Used packages: {len(used_packages)}")
    print(f"Unused packages: {len(unused_packages)}")
    print(f"Dev packages in main code: {len(dev_used_in_main)}")
    
    # Check for transitive dependencies note
    print("\n## NOTE ON TRANSITIVE DEPENDENCIES:")
    print("-" * 40)
    print("Some packages marked as 'unused' might be required as dependencies")
    print("of other packages. Run 'pip show <package>' to check dependents.")
    print("\nCommon transitive dependencies that might appear unused:")
    print("- scipy (required by scikit-learn)")
    print("- requests (might be used by other packages)")
    print("- aiofiles (might be required by FastAPI)")


if __name__ == "__main__":
    main()