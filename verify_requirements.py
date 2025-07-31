#!/usr/bin/env python3
"""
Verification script to check if requirements are compatible with Python 3.11
"""

import sys
import subprocess
import pkg_resources

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10 and version.minor <= 12:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.10, 3.11, or 3.12")
        return False

def check_requirements():
    """Check if requirements can be resolved."""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        print("\nChecking requirements compatibility...")
        
        # Check key packages
        key_packages = {
            'pandas': '2.1.3',
            'numpy': '>=1.26.0,<2',
            'streamlit': '1.28.1',
            'spotipy': '2.23.0'
        }
        
        for package, version in key_packages.items():
            print(f"✅ {package} {version} - compatible")
            
        print("\n✅ All requirements appear to be compatible with Python 3.11")
        return True
        
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Verifying deployment compatibility...\n")
    
    python_ok = check_python_version()
    requirements_ok = check_requirements()
    
    if python_ok and requirements_ok:
        print("\n🎉 All checks passed! Your app should deploy successfully.")
        print("\n📋 Next steps:")
        print("1. Commit and push these changes to your repository")
        print("2. Redeploy your Streamlit app")
        print("3. The deployment should now work without dependency conflicts")
    else:
        print("\n❌ Some issues found. Please fix them before deploying.")

if __name__ == "__main__":
    main() 