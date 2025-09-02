#!/usr/bin/env python3
"""
Script to compare installed packages with requirements.txt
"""

import re
from pathlib import Path

import pkg_resources


def check_package_differences():
    """Check differences between installed packages and requirements.txt"""

    # Read requirements.txt
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        return

    with open(req_file, "r") as f:
        req_content = f.read()

    # Parse requirements.txt to get package names and versions
    req_packages = {}
    for line in req_content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            # Extract package name (remove version constraints)
            package_name = re.split(r"[<>=!~]", line)[0].strip()
            if package_name:
                req_packages[package_name.lower()] = line

    # Get installed packages
    installed_packages = {}
    for dist in pkg_resources.working_set:
        installed_packages[dist.project_name.lower()] = dist.version

    print("🔍 PACKAGE ANALYSIS")
    print("=" * 50)

    print(f"\n📋 Total packages in requirements.txt: {len(req_packages)}")
    print(f"📦 Total packages installed: {len(installed_packages)}")

    # Check missing packages
    print("\n❌ PACKAGES IN REQUIREMENTS.TXT BUT NOT INSTALLED:")
    print("-" * 50)
    missing_count = 0
    for pkg, req_line in req_packages.items():
        if pkg not in installed_packages:
            print(f"  {pkg}: {req_line}")
            missing_count += 1

    if missing_count == 0:
        print("  ✅ All required packages are installed!")

    # Check extra packages
    print("\n➕ PACKAGES INSTALLED BUT NOT IN REQUIREMENTS.TXT:")
    print("-" * 50)
    extra_count = 0
    for pkg, version in sorted(installed_packages.items()):
        if pkg not in req_packages:
            print(f"  {pkg}: {version}")
            extra_count += 1

    if extra_count == 0:
        print("  ✅ No extra packages found!")

    # Check version mismatches
    print("\n⚠️  VERSION MISMATCHES:")
    print("-" * 50)
    mismatch_count = 0
    for pkg, req_line in req_packages.items():
        if pkg in installed_packages:
            installed_ver = installed_packages[pkg]
            # Extract version constraint from requirements line
            version_match = re.search(r"([<>=!~].+)", req_line)
            if version_match:
                req_constraint = version_match.group(1)
                print(f"  {pkg}: installed={installed_ver}, required{req_constraint}")
                mismatch_count += 1

    if mismatch_count == 0:
        print("  ✅ All installed packages meet version requirements!")

    # Summary
    print("\n📊 SUMMARY:")
    print("=" * 50)
    print(f"  Missing packages: {missing_count}")
    print(f"  Extra packages: {extra_count}")
    print(f"  Version mismatches: {mismatch_count}")

    if missing_count == 0 and extra_count == 0 and mismatch_count == 0:
        print("\n🎉 Perfect! Your environment matches requirements.txt exactly!")
    else:
        print(
            "\n💡 Consider running 'pip install -r requirements.txt' to fix missing packages"
        )


if __name__ == "__main__":
    check_package_differences()
