#!/usr/bin/env python3
"""
Test script to verify Shu-Ha-Ri phase advancement works correctly.
This simulates passing evaluations to trigger phase transitions.
"""

import sys
import os
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')

from continuous_teacher_v4 import (
    HierarchicalMasterySystem, 
    evaluate_response_shu,
    evaluate_response_ha, 
    evaluate_response_ri,
    TOPICS
)

def test_phase_advancement():
    """Test that phases advance correctly with passing scores."""
    print("=" * 70)
    print("Testing Shu-Ha-Ri Phase Advancement")
    print("=" * 70)
    
    # Initialize mastery system
    mastery_system = HierarchicalMasterySystem()
    
    # Test topic
    topic_name = 'Mathematics'
    
    print(f"\n📊 Initial status for {topic_name}:")
    status = mastery_system.get_current_status(topic_name)
    print(f"   Level: {status['current_level']}")
    print(f"   Phase: {status['phase']}")
    print(f"   Mastery: {status['mastery_percentage']:.1f}%")
    
    # Simulate multiple lessons and passing assessments to reach SHU → HA
    print(f"\n📚 Simulating lessons to reach HA phase...")
    
    # Need 70% to advance SHU → HA
    # Each lesson gives 15%, each passing assessment gives 20%
    # So: lesson (15) + pass (20) = 35% per cycle
    # Need 3 cycles to exceed 70%
    
    for i in range(4):
        print(f"\n--- Cycle {i+1} ---")
        
        # Record lesson
        result = mastery_system.record_lesson(topic_name, f"Test Lesson {i+1}")
        print(f"   Lesson: Mastery now {result['mastery_percentage']:.1f}%")
        
        # Simulate passing assessment (score 80+)
        result = mastery_system.record_assessment(topic_name, passed=True, score=85.0)
        print(f"   Assessment: Mastery now {result['mastery_percentage']:.1f}%")
        
        if result.get('phase_change'):
            change = result['phase_change']
            print(f"   🔔 {change['message']}")
    
    print(f"\n📊 Status after SHU phase:")
    status = mastery_system.get_current_status(topic_name)
    print(f"   Level: {status['current_level']}")
    print(f"   Phase: {status['phase']}")
    print(f"   Mastery: {status['mastery_percentage']:.1f}%")
    
    # Now simulate HA → RI transition (need 80%)
    if status['phase'] == 'ha':
        print(f"\n📚 Simulating lessons to reach RI phase...")
        
        for i in range(3):
            print(f"\n--- HA Cycle {i+1} ---")
            
            result = mastery_system.record_lesson(topic_name, f"HA Lesson {i+1}")
            print(f"   Lesson: Mastery now {result['mastery_percentage']:.1f}%")
            
            result = mastery_system.record_assessment(topic_name, passed=True, score=90.0)
            print(f"   Assessment: Mastery now {result['mastery_percentage']:.1f}%")
            
            if result.get('phase_change'):
                change = result['phase_change']
                print(f"   🔔 {change['message']}")
    
    print(f"\n📊 Status after HA phase:")
    status = mastery_system.get_current_status(topic_name)
    print(f"   Level: {status['current_level']}")
    print(f"   Phase: {status['phase']}")
    print(f"   Mastery: {status['mastery_percentage']:.1f}%")
    
    # Now simulate RI → Level 2 transition (need 90%)
    if status['phase'] == 'ri':
        print(f"\n📚 Simulating lessons to reach Level 2...")
        
        for i in range(2):
            print(f"\n--- RI Cycle {i+1} ---")
            
            result = mastery_system.record_lesson(topic_name, f"RI Lesson {i+1}")
            print(f"   Lesson: Mastery now {result['mastery_percentage']:.1f}%")
            
            result = mastery_system.record_assessment(topic_name, passed=True, score=95.0)
            print(f"   Assessment: Mastery now {result['mastery_percentage']:.1f}%")
            
            if result.get('phase_change'):
                change = result['phase_change']
                print(f"   🔔 {change['message']}")
    
    print(f"\n📊 Final status:")
    status = mastery_system.get_current_status(topic_name)
    print(f"   Level: {status['current_level']}")
    print(f"   Phase: {status['phase']}")
    print(f"   Mastery: {status['mastery_percentage']:.1f}%")
    
    # Show learning path
    path = mastery_system.get_learning_path(topic_name)
    print(f"\n🛤️  Learning Path for {topic_name}:")
    for step in path:
        marker = "👉" if step['is_current'] else "  "
        mastered = "✅" if step['is_mastered'] else "⬜"
        print(f"   {marker} Level {step['level']}: {step['phase'].upper():8} {step['mastery']:5.1f}% {mastered}")
    
    # Save state
    mastery_system.save_state()
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    return status['current_level'] == 2 and status['phase'] == 'shu'


def test_evaluation_functions():
    """Test the evaluation functions directly."""
    print("\n" + "=" * 70)
    print("Testing Evaluation Functions")
    print("=" * 70)
    
    # Test SHU evaluation
    question_data = {
        'question': 'What is the pattern in the Fibonacci sequence?',
        'keywords': ['sum', 'preceding', 'previous', 'add', 'sequence']
    }
    
    # Good response
    good_response = "The Fibonacci sequence is where each number is the sum of the two preceding numbers."
    passed, score, feedback = evaluate_response_shu(good_response, question_data)
    print(f"\nSHU Test (good response):")
    print(f"   Response: {good_response[:60]}...")
    print(f"   Passed: {passed}, Score: {score:.1f}%")
    print(f"   Feedback: {feedback}")
    
    # Bad response
    bad_response = "I think it has something to do with numbers."
    passed, score, feedback = evaluate_response_shu(bad_response, question_data)
    print(f"\nSHU Test (bad response):")
    print(f"   Response: {bad_response}")
    print(f"   Passed: {passed}, Score: {score:.1f}%")
    print(f"   Feedback: {feedback}")
    
    # Test HA evaluation
    ha_response = "The Fibonacci sequence works by adding the previous two numbers. For example, if we start with 0 and 1, the next number is 1. Then 1+1=2, 1+2=3, and so on. This pattern appears in nature, like in the arrangement of leaves on a stem. Therefore, we can use this sequence to model growth patterns."
    passed, score, feedback = evaluate_response_ha(ha_response, question_data, "Mathematics")
    print(f"\nHA Test:")
    print(f"   Response: {ha_response[:80]}...")
    print(f"   Passed: {passed}, Score: {score:.1f}%")
    print(f"   Feedback: {feedback}")
    
    # Test RI evaluation
    ri_response = "Imagine you're building a staircase. Each step is as tall as the two steps before it combined. That's Fibonacci! The key idea is simple addition, but the results are everywhere - from sunflower seeds to galaxy spirals. To remember it, just think: each number is the sum of the previous two. So 0, 1, 1, 2, 3, 5, 8... First you have nothing, then one, and each step builds on what came before."
    passed, score, feedback = evaluate_response_ri(ri_response, question_data, "Mathematics")
    print(f"\nRI Test:")
    print(f"   Response: {ri_response[:80]}...")
    print(f"   Passed: {passed}, Score: {score:.1f}%")
    print(f"   Feedback: {feedback}")


if __name__ == "__main__":
    # Test evaluation functions
    test_evaluation_functions()
    
    # Test phase advancement
    success = test_phase_advancement()
    
    if success:
        print("\n✅ All tests passed! Phase advancement works correctly.")
    else:
        print("\n❌ Tests failed. Phase advancement not working as expected.")
    
    sys.exit(0 if success else 1)
