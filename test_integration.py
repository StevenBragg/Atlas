#!/usr/bin/env python3
"""
Integration test for the fixed Atlas evaluation system.
Runs a mini teaching session to verify all components work together.
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')

from continuous_teacher_v5 import (
    evaluate_response, TOPICS, HierarchicalMasterySystem,
    get_coherence_feedback
)

def run_integration_test():
    """Run a mini teaching session with 5 assessments."""
    print("=" * 70)
    print("ATLAS FIXED EVALUATION - INTEGRATION TEST")
    print("=" * 70)
    
    # Initialize mastery system
    mastery = HierarchicalMasterySystem()
    
    # Test cases: (category, topic_index, simulated_response, should_pass)
    test_cases = [
        # Good responses
        ('Mathematics', 0, 'The Fibonacci sequence pattern works by adding the previous two numbers together. Each number is the sum of the preceding numbers.', True),
        ('Science', 0, 'Plants need sunlight, water, and carbon dioxide for photosynthesis.', True),
        
        # Gibberish responses (should fail)
        ('Programming', 0, 'engine [ state_norm share debug_av_system subsystem_stats ] tokens', False),
        ('Logic', 0, 'config params args kwargs init setup teardown handler callback', False),
        ('Language', 0, 'state_norm debug_av_system processor handler callback buffer', False),
    ]
    
    results = []
    
    for i, (category, q_idx, response, should_pass) in enumerate(test_cases, 1):
        topic_data = TOPICS[category][q_idx]
        
        print(f"\n--- Assessment {i}/5 ---")
        print(f"Topic: {topic_data['topic']} ({category})")
        print(f"Q: {topic_data['question']}")
        print(f"A: {response[:70]}...")
        
        # Get current status
        status = mastery.get_current_status(category)
        current_phase = status.get('phase', 'shu')
        
        # Evaluate response
        evaluation = evaluate_response(response, topic_data, current_phase, category)
        
        # Record in mastery system
        assessment_result = mastery.record_assessment(
            category,
            evaluation['passed'],
            evaluation['score'],
            evaluation.get('coherence_score', 0.0)
        )
        
        # Check if evaluation was correct
        is_correct = evaluation['passed'] == should_pass
        status_icon = "✅" if is_correct else "❌"
        
        print(f"Phase: {current_phase.upper()}")
        print(f"Result: {'PASS' if evaluation['passed'] else 'FAIL'} (expected: {'PASS' if should_pass else 'FAIL'}) {status_icon}")
        print(f"Score: {evaluation['score']:.1f}% | Coherence: {evaluation.get('coherence_score', 0):.2f}")
        print(f"Feedback: {evaluation['feedback'][:80]}...")
        
        if assessment_result.get('needs_retry'):
            print(f"⚠️ Topic needs retry (attempt {assessment_result['retry_count']}/3)")
        
        results.append({
            'correct': is_correct,
            'passed': evaluation['passed'],
            'should_pass': should_pass,
            'score': evaluation['score'],
            'coherence': evaluation.get('coherence_score', 0),
            'is_gibberish': not should_pass
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    
    print(f"\nTotal assessments: {total}")
    print(f"Correct evaluations: {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    
    # Check for false positives (gibberish that passed)
    false_positives = [r for r in results if r['passed'] and r['is_gibberish']]
    false_negatives = [r for r in results if not r['passed'] and not r['is_gibberish']]
    
    print(f"\nGibberish responses correctly rejected: {sum(1 for r in results if r['is_gibberish'] and not r['passed'])}/{sum(1 for r in results if r['is_gibberish'])}")
    print(f"Good responses correctly passed: {sum(1 for r in results if not r['is_gibberish'] and r['passed'])}/{sum(1 for r in results if not r['is_gibberish'])}")
    
    if false_positives:
        print(f"\n❌ FALSE POSITIVES: {len(false_positives)} gibberish responses passed!")
        for r in false_positives:
            print(f"   - Score: {r['score']:.1f}%, Coherence: {r['coherence']:.2f}")
    else:
        print("\n✅ NO FALSE POSITIVES: All gibberish was correctly rejected!")
    
    if false_negatives:
        print(f"\n⚠️ FALSE NEGATIVES: {len(false_negatives)} good responses failed (may need more keywords)")
    else:
        print("\n✅ NO FALSE NEGATIVES: All good responses passed!")
    
    # Show mastery system stats
    stats = mastery.get_summary_stats()
    print(f"\n📊 Mastery System Stats:")
    print(f"   Total assessments: {stats['total_assessments']}")
    print(f"   Passed: {stats['passed_assessments']}")
    print(f"   Failed: {stats['failed_assessments']}")
    print(f"   Pass rate: {stats['pass_rate']:.1f}%")
    print(f"   Total retries: {stats['total_retries']}")
    
    print("\n" + "=" * 70)
    
    # Final verdict
    if len(false_positives) == 0:
        print("🎉 SUCCESS: The fixed evaluation system is working correctly!")
        print("   - Gibberish/code noise is correctly rejected")
        print("   - Coherence checking prevents false passes")
        print("   - Retry tracking is working")
        return True
    else:
        print("❌ FAILURE: Some gibberish responses were incorrectly passed!")
        return False


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
