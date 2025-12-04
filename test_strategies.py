"""Test all strategies work with supervised signal."""
import sys
sys.path.insert(0, '.')

from self_organizing_av_system.core.curriculum_system import CurriculumSystem, CurriculumLevel
from self_organizing_av_system.core.learning_engine import LearningEngine
from self_organizing_av_system.core.meta_learning import LearningStrategy

# Initialize systems
curriculum = CurriculumSystem()
level_info = curriculum.CURRICULUM[CurriculumLevel.LEVEL_1_BASIC]
color_challenge_dict = level_info.challenges[1]  # "Recognize basic colors"

strategies = [
    LearningStrategy.HEBBIAN,
    LearningStrategy.COMPETITIVE,
    LearningStrategy.OJA,
    LearningStrategy.BCM,
]

print("Testing color recognition with different strategies...")
print("=" * 60)

for strategy in strategies:
    engine = LearningEngine(learning_rate=0.1)
    challenge = curriculum.create_challenge_object(color_challenge_dict)

    result = engine.execute_learning_loop(challenge, strategy=strategy)
    status = "PASS" if result.success else "FAIL"
    print(f"{strategy.name:15} -> {status}: {result.accuracy:.1%} in {result.iterations} iter")

print("=" * 60)
print("Done!")
