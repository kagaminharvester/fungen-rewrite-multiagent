#!/usr/bin/env python3
"""
Integration Verification Script

This script verifies that the FunGen integration is complete and all modules
can be imported correctly.

Usage:
    python verify_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def verify_imports():
    """Verify all critical imports."""
    results = []

    # Core modules
    try:
        from core.batch_processor import BatchProcessor
        from core.config import Config
        from core.frame_buffer import CircularFrameBuffer
        from core.model_manager import ModelManager
        from core.preprocessing import FramePreprocessor
        from core.video_processor import VideoProcessor

        results.append(("Core Modules", True, "All core modules imported successfully"))
    except Exception as e:
        results.append(("Core Modules", False, str(e)))

    # Tracker modules
    try:
        from trackers import ByteTracker
        from trackers.base_tracker import BaseTracker, Detection, Track
        from trackers.improved_tracker import ImprovedTracker

        results.append(("Tracker Modules", True, "All tracker modules imported successfully"))
    except Exception as e:
        results.append(("Tracker Modules", False, str(e)))

    # Utils modules
    try:
        from utils.conditional_imports import CUDA_AVAILABLE, TORCH_AVAILABLE
        from utils.performance import PerformanceMonitor
        from utils.platform_utils import detect_hardware, get_device

        results.append(("Utils Modules", True, "All utils modules imported successfully"))
    except Exception as e:
        results.append(("Utils Modules", False, str(e)))

    # UI modules (optional)
    try:
        from ui.agent_dashboard import AgentDashboard
        from ui.main_window import MainWindow

        results.append(("UI Modules", True, "All UI modules imported successfully"))
    except ImportError as e:
        results.append(("UI Modules", False, f"UI not available (expected on Pi): {e}"))
    except Exception as e:
        results.append(("UI Modules", False, str(e)))

    return results


def verify_files():
    """Verify critical files exist."""
    results = []

    files = [
        ("main.py", "Main entry point"),
        ("requirements.txt", "Dependencies"),
        ("setup.py", "Package installer"),
        ("README.md", "Documentation"),
        ("INTEGRATION_REPORT.md", "Integration report"),
        ("core/__init__.py", "Core package"),
        ("trackers/__init__.py", "Trackers package"),
        ("utils/__init__.py", "Utils package"),
    ]

    for file, desc in files:
        path = Path(__file__).parent / file
        if path.exists():
            results.append((desc, True, f"{file} exists"))
        else:
            results.append((desc, False, f"{file} NOT FOUND"))

    return results


def verify_config():
    """Verify configuration system."""
    results = []

    try:
        from core.config import PROFILES, Config

        # Check profiles exist
        expected_profiles = ["dev_pi", "prod_rtx3090", "debug"]
        for profile_name in expected_profiles:
            if profile_name in PROFILES:
                results.append((f"Profile: {profile_name}", True, "Profile exists"))
            else:
                results.append((f"Profile: {profile_name}", False, "Profile missing"))

        # Try auto-detect
        config = Config.auto_detect()
        results.append(("Config Auto-detect", True, f"Detected profile: {config.name}"))

    except Exception as e:
        results.append(("Configuration", False, str(e)))

    return results


def print_results(title, results):
    """Print verification results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    passed = 0
    failed = 0

    for name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {name:30} {message}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all verification checks."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    FunGen Integration Verification                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    all_passed = True

    # File verification
    file_results = verify_files()
    all_passed &= print_results("File Verification", file_results)

    # Import verification
    import_results = verify_imports()
    all_passed &= print_results("Import Verification", import_results)

    # Config verification
    config_results = verify_config()
    all_passed &= print_results("Configuration Verification", config_results)

    # Summary
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ VERIFICATION COMPLETE - All checks passed!")
        print("\nYou can now run:")
        print("  python main.py                    # GUI mode")
        print("  python main.py --cli video.mp4    # CLI mode")
    else:
        print("❌ VERIFICATION FAILED - Some checks did not pass")
        print("\nPlease review the errors above and:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Check for import errors")
        print("  3. Verify file structure")
    print(f"{'='*80}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
