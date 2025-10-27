#!/usr/bin/env python3
"""
Setup script for automated demo recording system
Installs dependencies and validates environment
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and report result"""
    print(f"  • {description}...", end=" ", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅")
            return True
        else:
            print("❌")
            if result.stderr:
                print(f"    Error: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ (timeout)")
        return False
    except Exception as e:
        print(f"❌ ({e})")
        return False


def main():
    """Main setup routine"""
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║    🎬 DEMO RECORDING SYSTEM - SETUP & VALIDATION          ║
╚═══════════════════════════════════════════════════════════╝

This script will:
1. Check Python version
2. Install required packages
3. Install Playwright browsers
4. Validate file structure
5. Run test recording (optional)

    """)
    
    # 1. Check Python version
    print("1️⃣  Python Environment")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"  • Python version: {python_version}", end="")
    if sys.version_info >= (3, 8):
        print(" ✅")
    else:
        print(" ❌ (need 3.8+)")
        return 1
    
    # 2. Install packages
    print("\n2️⃣  Installing Packages")
    packages_to_install = [
        (["pip", "install", "-q", "playwright"], "Playwright"),
        (["pip", "install", "-q", "pyyaml"], "PyYAML"),
    ]
    
    all_installed = True
    for cmd, name in packages_to_install:
        if not run_command(cmd, name):
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Some packages failed to install")
        return 1
    
    # 3. Install Playwright browsers
    print("\n3️⃣  Playwright Browsers")
    if not run_command(["playwright", "install"], "Install browsers"):
        print("⚠️  Browser installation had issues (may still work)")
    
    # 4. Validate files
    print("\n4️⃣  File Structure")
    required_files = [
        "slides_config.yaml",
        "SLIDES_CONFIG_LOADER.py",
        "AUTO_DEMO_RECORDER.py",
        "demo_slides.html",
    ]
    
    files_present = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"  • {file}: {status}")
        if not exists and file != "demo_slides.html":
            files_present = False
    
    if not files_present:
        print("\n❌ Missing required files!")
        return 1
    
    if not Path("demo_slides.html").exists():
        print("\n💡 Generating demo_slides.html...")
        run_command(["python", "SLIDES_CONFIG_LOADER.py"], "Generate HTML")
    
    # 5. Optional test
    print("\n5️⃣  Test Recording (Optional)")
    test_choice = input("  • Run test recording of first 3 slides? (y/N): ").strip().lower()
    
    if test_choice == 'y':
        print("\n  Starting test recording...")
        print("  (This will take 1-2 minutes)\n")
        cmd = ["python", "AUTO_DEMO_RECORDER.py", "--max-slides", "3", "--headless"]
        subprocess.run(cmd)
    
    # Summary
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                    ✅ SETUP COMPLETE                      ║
╚═══════════════════════════════════════════════════════════╝

🚀 Next Steps:

1. Generate HTML from YAML config:
   python SLIDES_CONFIG_LOADER.py

2. Record full demo video:
   python AUTO_DEMO_RECORDER.py

3. Edit slides in browser:
   Open demo_slides.html

Need Help?
  • Read: RECORDING_SYSTEM_README.md
  • View: demo_slides.html
  • Edit: slides_config.yaml

    """)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Setup cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
