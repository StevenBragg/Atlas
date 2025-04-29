#!/usr/bin/env python3
"""
Simple test runner for the full pipeline tests
"""

import unittest
import sys
import os

# Add parent dir to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import tests
from test_full_pipeline import TestFullPipeline

if __name__ == "__main__":
    print("Running full pipeline tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(TestFullPipeline("test_full_pipeline_with_resizing"))
    suite.addTest(TestFullPipeline("test_action_consistency_after_resize"))
    suite.addTest(TestFullPipeline("test_direct_pixel_output_consistency"))
    suite.addTest(TestFullPipeline("test_temporal_prediction_after_resize"))
    suite.addTest(TestFullPipeline("test_pipeline_robustness_with_multiple_resizes"))
    suite.addTest(TestFullPipeline("test_adaptability_after_resize"))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\nTest Summary:")
    print(f"  Ran {result.testsRun} tests")
    print(f"  Successes: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    # Exit with success if all tests pass
    sys.exit(not result.wasSuccessful()) 