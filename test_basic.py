#!/usr/bin/env python3
"""
Basic test script for the Bias-Resistant Music Recommendation System
Tests core functionality without requiring all dependencies.
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """Test that all required files and directories exist."""
    print("🔍 Testing project structure...")
    
    required_files = [
        "app.py",
        "demo.py", 
        "requirements.txt",
        "README.md",
        "SETUP_GUIDE.md",
        ".env.example"
    ]
    
    required_dirs = [
        "src",
        "src/ml",
        "src/spotify", 
        "src/ui",
        "src/utils"
    ]
    
    required_init_files = [
        "src/__init__.py",
        "src/ml/__init__.py",
        "src/spotify/__init__.py",
        "src/ui/__init__.py",
        "src/utils/__init__.py"
    ]
    
    all_good = True
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            all_good = False
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            all_good = False
    
    # Check __init__.py files
    for init_file in required_init_files:
        if os.path.exists(init_file):
            print(f"✅ {init_file}")
        else:
            print(f"❌ {init_file} - Missing")
            all_good = False
    
    return all_good

def test_python_imports():
    """Test basic Python imports."""
    print("\n🐍 Testing Python imports...")
    
    try:
        import json
        print("✅ json")
    except ImportError:
        print("❌ json")
        return False
    
    try:
        import logging
        print("✅ logging")
    except ImportError:
        print("❌ logging")
        return False
    
    try:
        import datetime
        print("✅ datetime")
    except ImportError:
        print("❌ datetime")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️ Testing configuration loading...")
    
    try:
        # Test if we can read the .env.example file
        if os.path.exists(".env.example"):
            with open(".env.example", "r", encoding="utf-8") as f:
                content = f.read()
                if "SPOTIFY_CLIENT_ID" in content:
                    print("✅ .env.example contains required variables")
                else:
                    print("❌ .env.example missing required variables")
                    return False
        else:
            print("❌ .env.example file not found")
            return False
        
        # Test if we can read requirements.txt
        if os.path.exists("requirements.txt"):
            with open("requirements.txt", "r", encoding="utf-8") as f:
                content = f.read()
                if "streamlit" in content and "spotipy" in content:
                    print("✅ requirements.txt contains required packages")
                else:
                    print("❌ requirements.txt missing required packages")
                    return False
        else:
            print("❌ requirements.txt file not found")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def test_app_structure():
    """Test the main app structure."""
    print("\n📱 Testing app structure...")
    
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check for key imports
        required_imports = [
            "streamlit",
            "SpotifyClient",
            "SpotifyDataProcessor", 
            "HybridRecommendationModel"
        ]
        
        for imp in required_imports:
            if imp in content:
                print(f"✅ {imp} import found")
            else:
                print(f"❌ {imp} import missing")
                return False
        
        # Check for main function
        if "def main():" in content:
            print("✅ main() function found")
        else:
            print("❌ main() function missing")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error testing app structure: {e}")
        return False

def test_demo_script():
    """Test the demo script structure."""
    print("\n🎬 Testing demo script...")
    
    try:
        with open("demo.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check for key functions
        required_functions = [
            "def main():",
            "def demo_authentication():",
            "def demo_data_loading(",
            "def demo_recommendation_generation("
        ]
        
        for func in required_functions:
            if func in content:
                print(f"✅ {func} found")
            else:
                print(f"❌ {func} missing")
                return False
                
        return True
    except Exception as e:
        print(f"❌ Error testing demo script: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Basic Tests for Music Recommendation System")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Imports", test_python_imports),
        ("Configuration", test_config_loading),
        ("App Structure", test_app_structure),
        ("Demo Script", test_demo_script)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application structure is ready.")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Set up Spotify API credentials in .env file")
        print("   3. Run the app: streamlit run app.py")
        print("   4. Or run the demo: python demo.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 