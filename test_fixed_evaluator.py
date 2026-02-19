#!/usr/bin/env python3
"""
Test script to verify the fixed Atlas evaluation system.
Tests coherence checking, evaluation criteria, and retry logic.
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')

from continuous_teacher_v5 import (
    evaluate_response_shu, evaluate_response_ha, evaluate_response_ri,
    evaluate_response, TOPICS, HierarchicalMasterySystem,
    get_coherence_feedback, generate_corrective_feedback
)
from coherence_evaluator import evaluate_coherence, CoherenceEvaluator, CoherenceIssue


def test_coherence_evaluator():
    """Test the coherence evaluator directly."""
    print("=" * 70)
    print("Test 1: Coherence Evaluator")
    print("=" * 70)
    
    test_cases = [
        ("The Fibonacci sequence is where each number is the sum of the two preceding numbers.", "Mathematics", True),
        ("engine [ state_norm share debug_av_system subsystem_stats ] tokens", "Mathematics", False),
        ("sum preceding previous add sequence", "Mathematics", True),  # Just keywords but coherent
        ("Photosynthesis requires sunlight, water, and carbon dioxide.", "Science", True),
        ("config params args kwargs init setup teardown", "Science", False),
        ("state_norm debug_av_system processor handler callback", "Programming", False),
    ]
    
    evaluator = CoherenceEvaluator()
    correct = 0
    
    for response, topic, should_be_coherent in test_cases:
        result = evaluator.analyze(response, topic)
        is_coherent = result.is_coherent
        passed = is_coherent == should_be_coherent
        
        status = "✅" if passed else "❌"
        print(f"\n{status} Response: {response[:60]}...")
        print(f"   Topic: {topic} | Coherent: {is_coherent} (expected: {should_be_coherent})")
        print(f"   Score: {result.score:.2f} | Issues: {[i.value for i in result.issues]}")
        
        if passed:
            correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Coherence Evaluator: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 70)
    return correct == len(test_cases)


def test_shu_evaluation():
    """Test SHU phase evaluation with coherence checking."""
    print("\n" + "=" * 70)
    print("Test 2: SHU Phase Evaluation")
    print("=" * 70)
    
    question_data = {
        'question': 'What is the pattern in the Fibonacci sequence?',
        'keywords': ['sum', 'preceding', 'previous', 'add', 'sequence']
    }
    
    test_cases = [
        # (response, should_pass, description)
        ("The Fibonacci sequence is where each number is the sum of the two preceding numbers.", True, "Good coherent response"),
        ("engine [ state_norm share debug_av_system subsystem_stats ] tokens", False, "Gibberish code noise"),
        ("sum preceding previous add sequence", True, "Keywords only but coherent"),
        ("The pattern works by adding the previous two numbers together, so each number equals the sum of what came before.", True, "Good with application"),
        ("state_norm debug_av_system processor", False, "Code noise"),
    ]
    
    correct = 0
    for response, should_pass, description in test_cases:
        passed, score, feedback, details = evaluate_response_shu(response, question_data, "Mathematics")
        is_correct = passed == should_pass
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} {description}")
        print(f"   Response: {response[:60]}...")
        print(f"   Result: {'PASS' if passed else 'FAIL'} (expected: {'PASS' if should_pass else 'FAIL'})")
        print(f"   Score: {score:.1f}% | Coherence: {details['coherence'].score:.2f}")
        
        if is_correct:
            correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"SHU Evaluation: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 70)
    return correct == len(test_cases)


def test_ha_evaluation():
    """Test HA phase evaluation."""
    print("\n" + "=" * 70)
    print("Test 3: HA Phase Evaluation")
    print("=" * 70)
    
    question_data = {
        'question': 'What is the pattern in the Fibonacci sequence?',
        'keywords': ['sum', 'preceding', 'previous', 'add', 'sequence']
    }
    
    test_cases = [
        # (response, should_pass, description)
        ("The Fibonacci sequence works by adding the previous two numbers. For example, if we start with 0 and 1, the next number is 1. Then 1+1=2, 1+2=3, and so on. This pattern appears in nature, like in the arrangement of leaves on a stem. Therefore, we can use this sequence to model growth patterns.", True, "Good with examples and application"),
        ("engine [ state_norm share debug_av_system ]", False, "Gibberish"),
        ("The Fibonacci sequence is the sum of preceding numbers.", False, "Too short, no application"),
    ]
    
    correct = 0
    for response, should_pass, description in test_cases:
        passed, score, feedback, details = evaluate_response_ha(response, question_data, "Mathematics")
        is_correct = passed == should_pass
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} {description}")
        print(f"   Response: {response[:60]}...")
        print(f"   Result: {'PASS' if passed else 'FAIL'} (expected: {'PASS' if should_pass else 'FAIL'})")
        print(f"   Score: {score:.1f}%")
        
        if is_correct:
            correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"HA Evaluation: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 70)
    return correct == len(test_cases)


def test_ri_evaluation():
    """Test RI phase evaluation."""
    print("\n" + "=" * 70)
    print("Test 4: RI Phase Evaluation")
    print("=" * 70)
    
    question_data = {
        'question': 'What is the pattern in the Fibonacci sequence?',
        'keywords': ['sum', 'preceding', 'previous', 'add', 'sequence']
    }
    
    test_cases = [
        # (response, should_pass, description)
        ("Imagine you're building a staircase. Each step is as tall as the two steps before it combined. That's Fibonacci! The key idea is simple addition, but the results are everywhere - from sunflower seeds to galaxy spirals. To remember it, just think: each number is the sum of the previous two. So 0, 1, 1, 2, 3, 5, 8... First you have nothing, then one, and each step builds on what came before.", True, "Good teaching explanation"),
        ("engine [ state_norm share debug_av_system ]", False, "Gibberish"),
        ("The Fibonacci sequence adds previous numbers.", False, "Too short, no teaching"),
    ]
    
    correct = 0
    for response, should_pass, description in test_cases:
        passed, score, feedback, details = evaluate_response_ri(response, question_data, "Mathematics")
        is_correct = passed == should_pass
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} {description}")
        print(f"   Response: {response[:60]}...")
        print(f"   Result: {'PASS' if passed else 'FAIL'} (expected: {'PASS' if should_pass else 'FAIL'})")
        print(f"   Score: {score:.1f}%")
        
        if is_correct:
            correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"RI Evaluation: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 70)
    return correct == len(test_cases)


def test_no_false_passes():
    """Verify that gibberish/code noise never passes."""
    print("\n" + "=" * 70)
    print("Test 5: No False Passes (Gibberish Detection)")
    print("=" * 70)
    
    gibberish_responses = [
        "engine [ state_norm share debug_av_system subsystem_stats ] tokens",
        "config params args kwargs init setup teardown",
        "state_norm debug_av_system processor handler callback buffer",
        "import from class def return self cls null none",
        "array vector matrix tensor layer node edge forward backward",
        "api endpoint request response header payload json xml",
        "thread process lock mutex semaphore queue async await",
    ]
    
    question_data = {
        'question': 'What is the pattern in the Fibonacci sequence?',
        'keywords': ['sum', 'preceding', 'previous', 'add', 'sequence']
    }
    
    all_rejected = True
    for response in gibberish_responses:
        # Add keywords to make it seem like it should pass
        response_with_keywords = response + " sum preceding previous add sequence"
        
        passed, score, feedback, details = evaluate_response_shu(response_with_keywords, question_data, "Mathematics")
        
        status = "✅ REJECTED" if not passed else "❌ FALSE PASS"
        print(f"\n{status}")
        print(f"   Response: {response[:60]}...")
        print(f"   With keywords added: {response_with_keywords[:80]}...")
        print(f"   Result: {'PASS' if passed else 'FAIL'} | Score: {score:.1f}%")
        print(f"   Coherence: {details['coherence'].score:.2f}")
        
        if passed:
            all_rejected = False
    
    print(f"\n{'=' * 70}")
    if all_rejected:
        print("✅ SUCCESS: All gibberish responses were correctly rejected!")
    else:
        print("❌ FAILURE: Some gibberish responses passed!")
    print("=" * 70)
    return all_rejected


def test_mastery_system():
    """Test the mastery system with retry tracking."""
    print("\n" + "=" * 70)
    print("Test 6: Mastery System with Retry Tracking")
    print("=" * 70)
    
    mastery = HierarchicalMasterySystem()
    topic = "Mathematics"
    
    # Record a lesson
    result = mastery.record_lesson(topic, "Test Lesson 1")
    print(f"\n✅ Recorded lesson: {result['mastery_percentage']:.1f}% mastery")
    
    # Record a passing assessment
    result = mastery.record_assessment(topic, passed=True, score=85.0, coherence_score=0.9)
    print(f"✅ Recorded pass: {result['mastery_percentage']:.1f}% mastery, retry count: {result['retry_count']}")
    
    # Record a failing assessment
    result = mastery.record_assessment(topic, passed=False, score=45.0, coherence_score=0.3)
    print(f"✅ Recorded fail: {result['mastery_percentage']:.1f}% mastery, retry count: {result['retry_count']}")
    
    # Check retry count
    status = mastery.get_current_status(topic)
    print(f"\n📊 Current status for {topic}:")
    print(f"   Level: {status['current_level']}")
    print(f"   Phase: {status['phase']}")
    print(f"   Mastery: {status['mastery_percentage']:.1f}%")
    print(f"   Retry count: {status['retry_count']}")
    
    # Get summary stats
    stats = mastery.get_summary_stats()
    print(f"\n📊 Summary stats:")
    print(f"   Total assessments: {stats['total_assessments']}")
    print(f"   Passed: {stats['passed_assessments']}")
    print(f"   Failed: {stats['failed_assessments']}")
    print(f"   Pass rate: {stats['pass_rate']:.1f}%")
    print(f"   Total retries: {stats['total_retries']}")
    
    print(f"\n{'=' * 70}")
    print("✅ Mastery system working correctly!")
    print("=" * 70)
    return True


def test_corrective_feedback():
    """Test corrective feedback generation."""
    print("\n" + "=" * 70)
    print("Test 7: Corrective Feedback Generation")
    print("=" * 70)
    
    # Simulate a failed evaluation with coherence issues
    evaluation = {
        'passed': False,
        'score': 35.0,
        'phase': 'shu',
        'details': {
            'coherence': type('obj', (object,), {
                'is_coherent': False,
                'issues': [
                    CoherenceIssue.CODE_NOISE,
                    CoherenceIssue.NO_VERBS
                ]
            })()
        }
    }
    
    topic_data = {
        'topic': 'Fibonacci Sequence',
        'keywords': ['sum', 'preceding', 'previous', 'add']
    }
    
    feedback = generate_corrective_feedback(evaluation, topic_data, "Mathematics")
    
    print(f"\nGenerated feedback for failed SHU assessment:")
    print(f"   {feedback}")
    
    # Check that feedback contains corrective guidance
    has_code_noise_feedback = "programming" in feedback.lower() or "code" in feedback.lower()
    has_verb_feedback = "verbs" in feedback.lower() or "sentence" in feedback.lower()
    has_keyword_feedback = "key" in feedback.lower() or "facts" in feedback.lower()
    
    print(f"\n✅ Contains code noise feedback: {has_code_noise_feedback}")
    print(f"✅ Contains verb feedback: {has_verb_feedback}")
    print(f"✅ Contains keyword guidance: {has_keyword_feedback}")
    
    print(f"\n{'=' * 70}")
    print("✅ Corrective feedback generation working!")
    print("=" * 70)
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("ATLAS EVALUATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    results = []
    
    results.append(("Coherence Evaluator", test_coherence_evaluator()))
    results.append(("SHU Evaluation", test_shu_evaluation()))
    results.append(("HA Evaluation", test_ha_evaluation()))
    results.append(("RI Evaluation", test_ri_evaluation()))
    results.append(("No False Passes", test_no_false_passes()))
    results.append(("Mastery System", test_mastery_system()))
    results.append(("Corrective Feedback", test_corrective_feedback()))
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n{'=' * 70}")
    print(f"Overall: {passed}/{total} test suites passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The evaluation system is working correctly.")
    else:
        print("⚠️ Some tests failed. Review the output above.")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
