#!/usr/bin/env python3
"""
Atlas Telegram Reporter - v3 Fixed Version

Reads from:
- shared_brain.pkl (source of truth for vocabulary)
- session_stats.json (persistent session tracking)
- conversations.json (Q&A history)
"""

import os
import sys
import pickle
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Ensure we use the Atlas directory for imports and data
ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
sys.path.insert(0, str(ATLAS_DIR / 'self_organizing_av_system'))

# Import assessment history tracker
from assessment_history_tracker import get_tracker, AssessmentHistoryTracker

def get_git_commits():
    """Get recent git commits"""
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '-5'],
            capture_output=True,
            text=True,
            cwd=ATLAS_DIR
        )
        return result.stdout.strip().split('\n') if result.stdout else []
    except Exception as e:
        print(f"Git error: {e}")
        return []

def get_improvements():
    """Get recent improvement files"""
    imp_dir = ATLAS_DIR / 'improvements'
    if not imp_dir.exists():
        return []
    
    try:
        files = sorted(imp_dir.glob('improvement_*.py'), 
                       key=lambda p: p.stat().st_mtime, 
                       reverse=True)[:3]
        return [f.name for f in files]
    except Exception as e:
        print(f"Improvements error: {e}")
        return []

def get_assessment_history():
    """Get recent assessments from history tracker"""
    try:
        tracker = AssessmentHistoryTracker()
        recent = tracker.get_recent_assessments(5)
        topics = tracker.get_all_topics()
        stats = tracker.get_summary_stats()
        return {
            'recent': recent,
            'topics': topics,
            'stats': stats
        }
    except Exception as e:
        print(f"Assessment history error: {e}")
        return {'recent': [], 'topics': [], 'stats': {}}

def get_active_agents():
    """Check running processes - look for continuous teacher"""
    try:
        # Check for continuous teacher (single source of truth)
        result = subprocess.run(['pgrep', '-f', 'continuous_teacher.py'], 
                              capture_output=True, text=True)
        teachers = [p for p in result.stdout.strip().split('\n') if p]
        
        # Also check for the shell script
        result2 = subprocess.run(['pgrep', '-f', 'run_continuous_teacher.sh'], 
                              capture_output=True, text=True)
        scripts = [p for p in result2.stdout.strip().split('\n') if p]
        
        # Also check for autonomous agents
        result3 = subprocess.run(['pgrep', '-f', 'autonomous'], 
                              capture_output=True, text=True)
        autonomous = [p for p in result3.stdout.strip().split('\n') if p]
        
        total = len(teachers) + len(scripts) + len(autonomous)
        
        # Return detailed info
        return {
            'count': total,
            'teachers': len(teachers),
            'scripts': len(scripts),
            'autonomous': len(autonomous),
            'teacher_pids': teachers,
            'script_pids': scripts
        }
    except Exception as e:
        print(f"Process check error: {e}")
        return {'count': 0, 'teachers': 0, 'scripts': 0, 'autonomous': 0, 'teacher_pids': [], 'script_pids': []}

def get_session_stats():
    """Get persistent session statistics"""
    stats_file = ATLAS_DIR / 'teacher_state' / 'session_stats.json'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Session stats error: {e}")
    
    return {
        'total_lessons': 0,
        'total_questions': 0,
        'total_conversations': 0,
        'sessions_completed': 0,
        'last_session_time': None
    }

def get_conversations():
    """Get recent conversations from persistent storage"""
    conv_file = ATLAS_DIR / 'teacher_state' / 'conversations.json'
    if conv_file.exists():
        try:
            with open(conv_file, 'r') as f:
                conversations = json.load(f)
            # Return last 3 conversations
            return conversations[-3:]
        except Exception as e:
            print(f"Conversations error: {e}")
    return []

def get_teacher_log_stats():
    """Get stats from teacher log file"""
    log_file = ATLAS_DIR / 'logs' / 'continuous_teacher.log'
    
    stats = {
        'last_session_time': None,
        'is_stale': True,
        'recent_lines': []
    }
    
    if not log_file.exists():
        return stats
    
    try:
        # Check log file age
        log_mtime = log_file.stat().st_mtime
        stats['last_session_time'] = datetime.fromtimestamp(log_mtime)
        
        # Consider log stale if older than 30 minutes
        time_since_update = datetime.now() - stats['last_session_time']
        stats['is_stale'] = time_since_update.total_seconds() > 1800  # 30 min
        
        # Read last 50 lines
        with open(log_file) as f:
            lines = f.readlines()
        
        stats['recent_lines'] = lines[-50:]
        
    except Exception as e:
        print(f"Log read error: {e}")
    
    return stats

def get_atlas_stats():
    """Get current Atlas learning stats - reads directly from shared_brain.pkl"""
    stats = {
        'vocabulary': 0,
        'total_tokens': 0,
        'unique_contexts': 0,
        'brain_file_age': None,
        'brain_file_age_minutes': 0,
        
        # Session stats
        'total_lessons': 0,
        'total_questions': 0,
        'total_conversations': 0,
        'sessions_completed': 0,
        'last_session_time': None,
        
        # Process stats
        'active_agents': 0,
        'teacher_processes': 0,
        'script_processes': 0,
        'autonomous_processes': 0,
        
        # Conversations
        'conversations': [],
        
        # Other
        'improvements': [],
        'commits': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'teacher_log_stale': True
    }
    
    # Load from shared brain pickle (the actual source of truth)
    brain_path = ATLAS_DIR / 'shared_brain.pkl'
    if brain_path.exists():
        try:
            # Get file age
            mtime = brain_path.stat().st_mtime
            age_seconds = datetime.now().timestamp() - mtime
            stats['brain_file_age'] = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            stats['brain_file_age_minutes'] = int(age_seconds / 60)
            
            # Load the brain state
            with open(brain_path, 'rb') as f:
                state = pickle.load(f)
            
            # Extract vocabulary count from token_to_idx
            token_to_idx = state.get('token_to_idx', {})
            stats['vocabulary'] = len(token_to_idx)
            
            # Get total tokens processed
            stats['total_tokens'] = state.get('total_tokens', 0)
            
            # Get unique contexts
            context_weights = state.get('context_weights', {})
            stats['unique_contexts'] = len(context_weights)
            
            print(f"✓ Loaded brain: {stats['vocabulary']} vocab, {stats['total_tokens']} tokens")
            
        except Exception as e:
            print(f"Error loading brain: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ Brain file not found: {brain_path}")
    
    # Get session stats from persistent storage
    session_stats = get_session_stats()
    stats['total_lessons'] = session_stats.get('total_lessons', 0)
    stats['total_questions'] = session_stats.get('total_questions', 0)
    stats['total_conversations'] = session_stats.get('total_conversations', 0)
    stats['sessions_completed'] = session_stats.get('sessions_completed', 0)
    stats['last_session_time'] = session_stats.get('last_session_time')
    
    # Get conversations from persistent storage
    stats['conversations'] = get_conversations()
    
    # Get active agents
    agent_info = get_active_agents()
    stats['active_agents'] = agent_info['count']
    stats['teacher_processes'] = agent_info['teachers']
    stats['script_processes'] = agent_info['scripts']
    stats['autonomous_processes'] = agent_info['autonomous']
    stats['teacher_pids'] = agent_info.get('teacher_pids', [])
    stats['script_pids'] = agent_info.get('script_pids', [])
    
    # Get teacher log stats
    log_stats = get_teacher_log_stats()
    stats['teacher_log_stale'] = log_stats['is_stale']
    stats['teacher_log_last_update'] = log_stats['last_session_time'].strftime('%H:%M') if log_stats['last_session_time'] else 'unknown'
    
    # Get assessment history
    assessment_data = get_assessment_history()
    stats['assessment_history'] = assessment_data
    stats['recent_assessments'] = assessment_data.get('recent', [])
    stats['assessment_topics'] = assessment_data.get('topics', [])
    stats['assessment_stats'] = assessment_data.get('stats', {})
    
    # Get improvements
    stats['improvements'] = get_improvements()
    
    # Get commits
    stats['commits'] = get_git_commits()[:3]
    
    return stats

def format_report(stats):
    """Format detailed stats for Telegram"""
    
    # Determine status emoji based on brain file age
    age_minutes = stats.get('brain_file_age_minutes', 0)
    if age_minutes < 5:
        status_emoji = "🟢"
    elif age_minutes < 30:
        status_emoji = "🟡"
    else:
        status_emoji = "🔴"
    
    report = f"""🧠 *Atlas Learning Update* {status_emoji}

📊 *Brain Stats (from shared_brain.pkl):*
• Vocabulary: *{stats['vocabulary']:,}* words
• Total tokens: {stats['total_tokens']:,}
• Unique contexts: {stats['unique_contexts']:,}
• Brain updated: {stats.get('brain_file_age', 'unknown')} ({age_minutes}m ago)

📚 *Teaching Statistics:*
• Total lessons: {stats['total_lessons']}
• Total questions: {stats['total_questions']}
• Sessions completed: {stats['sessions_completed']}

🤖 *Active Processes:*
• Total active: {stats['active_agents']}
  - Teacher processes: {stats['teacher_processes']}
  - Script runners: {stats['script_processes']}
  - Autonomous agents: {stats['autonomous_processes']}
"""
    
    # Show recent conversations if available
    if stats['conversations']:
        report += f"\n💬 *Recent Q\u0026A:*\n"
        for i, conv in enumerate(stats['conversations'][-3:], 1):
            time_str = conv.get('time', '??:??')
            topic = conv.get('topic', 'Unknown')
            q = conv['q'][:40] + '...' if len(conv['q']) > 40 else conv['q']
            a = conv['a'][:45] + '...' if len(conv['a']) > 45 else conv['a']
            report += f"\n{i}. 🕐 {time_str} | {topic}\n"
            report += f"   Q: _{q}_\n"
            report += f"   A: `{a}`\n"
    else:
        report += "\n💬 *Recent Q\u0026A:*\n"
        if stats.get('teacher_log_stale', True):
            last_update = stats.get('teacher_log_last_update', 'unknown')
            report += f"   _No recent sessions. Last activity: {last_update}_\n"
        else:
            report += "   No conversations recorded yet.\n"
    
    # Show recent assessments if available
    recent_assessments = stats.get('recent_assessments', [])
    if recent_assessments:
        report += f"\n📝 *Recent Assessments:*\n"
        for i, assessment in enumerate(recent_assessments, 1):
            from datetime import datetime
            ts = datetime.fromisoformat(assessment.timestamp).strftime('%m-%d %H:%M')
            status = "✅" if assessment.passed else "❌"
            report += f"{i}. {status} {assessment.topic} L{assessment.level}-{assessment.phase.upper()}\n"
            report += f"   Score: {assessment.score:.0f}% | Attempt {assessment.attempt_number} | {ts}\n"
        
        # Show current phase/level for active topics
        active_topics = stats.get('assessment_topics', [])
        if active_topics:
            report += f"\n🎯 *Current Status:*\n"
            tracker = AssessmentHistoryTracker()
            for topic in active_topics[:5]:  # Limit to 5 topics
                current = tracker.get_current_phase_for_topic(topic)
                if current:
                    status_emoji = "🟢" if current['last_passed'] else "🔴"
                    report += f"{status_emoji} {topic}: L{current['level']}-{current['phase'].upper()}\n"
        
        # Show assessment stats
        assessment_stats = stats.get('assessment_stats', {})
        if assessment_stats:
            report += f"\n📈 *Assessment Summary:*\n"
            report += f"• Total: {assessment_stats.get('total_assessments', 0)}\n"
            report += f"• Pass Rate: {assessment_stats.get('pass_rate', 0):.1f}%\n"
            report += f"• Avg Score: {assessment_stats.get('average_score', 0):.1f}%\n"
    else:
        report += f"\n📝 *Recent Assessments:*\n"
        report += "   No assessments recorded yet.\n"
    
    if stats['improvements']:
        report += f"\n🔧 *Recent Improvements:*\n"
        for imp in stats['improvements']:
            report += f"• {imp}\n"
    
    if stats['commits']:
        report += f"\n📦 *Recent Commits:*\n"
        for commit in stats['commits']:
            short = commit[:50] + '...' if len(commit) > 50 else commit
            report += f"• `{short}`\n"
    
    report += f"\n⏰ _Report generated: {stats['timestamp']}_"
    
    return report

def send_telegram_message(message):
    """Send message via Telegram bot"""
    import urllib.request
    import urllib.parse
    
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("⚠ Telegram credentials not set")
        print("="*60)
        print(message)
        print("="*60)
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }).encode()
    
    try:
        with urllib.request.urlopen(url, data=data, timeout=30) as response:
            print("✓ Message sent successfully")
            return True
    except Exception as e:
        print(f"✗ Failed to send: {e}")
        print("="*60)
        print(message)
        print("="*60)
        return False

def main():
    print(f"\n{'='*60}")
    print(f"Atlas Telegram Reporter v3 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    stats = get_atlas_stats()
    report = format_report(stats)
    
    print(f"\nStats collected:")
    print(f"  - Vocabulary: {stats['vocabulary']:,} words")
    print(f"  - Total tokens: {stats['total_tokens']:,}")
    print(f"  - Total lessons: {stats['total_lessons']}")
    print(f"  - Total questions: {stats['total_questions']}")
    print(f"  - Sessions completed: {stats['sessions_completed']}")
    print(f"  - Brain file age: {stats.get('brain_file_age', 'unknown')} ({stats.get('brain_file_age_minutes', 0)}m ago)")
    print(f"  - Active agents: {stats['active_agents']}")
    print(f"    - Teacher processes: {stats['teacher_processes']}")
    print(f"    - Script runners: {stats['script_processes']}")
    
    print(f"\nSending report...")
    success = send_telegram_message(report)
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}\n")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
