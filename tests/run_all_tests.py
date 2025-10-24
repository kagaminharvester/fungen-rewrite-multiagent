"""
Master test runner for FunGen rewrite test suite.

This script runs all unit tests and generates a comprehensive coverage report.
It works with or without pytest installed.

Author: test-engineer-1 agent
Date: 2025-10-24
Target: 80%+ code coverage
"""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test modules to run
TEST_MODULES = [
    "tests.unit.test_config",
    "tests.unit.test_conditional_imports",
    "tests.unit.test_platform_utils",
    "tests.unit.test_performance",
    "tests.unit.test_frame_buffer",
    "tests.unit.test_model_manager",
    "tests.unit.test_byte_tracker",
    "tests.unit.test_improved_tracker",
    "tests.unit.test_ui_components",
    "tests.unit.test_video_processor",
    "tests.unit.test_preprocessing",
    "tests.unit.test_kalman_filter",
    "tests.unit.test_optical_flow",
    "tests.unit.test_batch_processor",
]


def check_pytest_available():
    """Check if pytest is available."""
    try:
        import pytest

        return True
    except ImportError:
        return False


def run_with_pytest():
    """Run tests using pytest with coverage."""
    print("\n" + "=" * 80)
    print("Running tests with pytest")
    print("=" * 80 + "\n")

    cmd = [
        "python3",
        "-m",
        "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
    ]

    # Try to add coverage if available
    try:
        import pytest_cov

        cmd.extend(
            [
                "--cov=core",
                "--cov=trackers",
                "--cov=utils",
                "--cov=ui",
                "--cov-report=term-missing",
                "--cov-report=html:tests/coverage_html",
            ]
        )
    except ImportError:
        print("Warning: pytest-cov not available, running without coverage\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def run_without_pytest():
    """Run tests manually without pytest."""
    print("\n" + "=" * 80)
    print("Running tests without pytest (manual mode)")
    print("=" * 80 + "\n")

    total_passed = 0
    total_failed = 0
    test_results = []

    for module_name in TEST_MODULES:
        print(f"\n{'=' * 80}")
        print(f"Running {module_name}")
        print("=" * 80)

        try:
            # Import module
            module_path = module_name.replace(".", "/")
            test_file = PROJECT_ROOT / f"{module_path}.py"

            if not test_file.exists():
                print(f"Warning: Test file not found: {test_file}")
                continue

            # Import and run
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Run tests if run_all_tests exists
            if hasattr(module, "run_all_tests"):
                start_time = time.time()
                success = module.run_all_tests()
                elapsed = time.time() - start_time

                test_results.append({"module": module_name, "success": success, "time": elapsed})

                if success:
                    total_passed += 1
                else:
                    total_failed += 1
            else:
                print(f"Warning: {module_name} has no run_all_tests() function")

        except Exception as e:
            print(f"ERROR running {module_name}: {e}")
            total_failed += 1
            test_results.append({"module": module_name, "success": False, "error": str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)

    for result in test_results:
        status = "PASSED" if result["success"] else "FAILED"
        time_str = f"({result.get('time', 0):.2f}s)" if "time" in result else ""
        print(f"{result['module']}: {status} {time_str}")

        if "error" in result:
            print(f"  Error: {result['error']}")

    print("\n" + "=" * 80)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 80 + "\n")

    return total_failed == 0


def main():
    """Main test runner."""
    print("\n" + "=" * 80)
    print("FunGen Rewrite Test Suite")
    print("Author: test-engineer-1 agent")
    print("Date: 2025-10-24")
    print("Target: 80%+ code coverage")
    print("=" * 80)

    pytest_available = check_pytest_available()

    if pytest_available:
        print("\nPytest detected - using pytest for test execution")
        success = run_with_pytest()
    else:
        print("\nPytest not available - using manual test execution")
        success = run_without_pytest()

    if success:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80 + "\n")
        return 0
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
