#!/usr/bin/env python3
"""
Atlas Telegram Reporter - Fixed Version

Sends accurate real-time updates about Atlas's learning progress to Telegram.
Reads directly from shared_brain.pkl (source of truth) for fresh data.
"""

import os
import sys
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure we use the Atlas directory for imports and data
ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
sys.path.insert(0, str(ATLAS_DIR / 'self_organizing_av_system'))

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

def get_active_agents():
    """Check running processes"""
    try:
        # Check for continuous teacher
        result = subprocess.run(['pgrep', '-f', 'continuous_teacher'], 
                              capture_output=True, text=True)
        count = len([p for p in result.stdout.strip().split('\n') if p]) if result.stdout else 0
        
        # Also check for autonomous agents
        result2 = subprocess.run(['pgrep', '-f', 'autonomous'], 
                              capture_output=True, text=True)
        count += len([p for p in result2.stdout.strip().split('\n') if p]) if result2.stdout else 0
        
        return count
    except:
        return 0

def get_teacher_stats():
    """Get stats from teacher logs - check both autonomous and continuous"""
    
    # Check continuous teacher log first (active)
    continuous_log = ATLAS_DIR / 'logs/continuous_teacher.log'
    teacher_log = ATLAS_DIR / 'logs/teacher_agent.log'
    
    stats = {
        'lessons_taught': 0,
        'questions_asked': 0,
        'concepts_explained': 0,
        'recent_conversations': [],
        'last_log_time': None,
        'is_stale': True
    }
    
    # Use continuous log if it exists and is recent
    log_file = continuous_log if continuous_log.exists() else teacher_log
    
    if not log_file.exists():
        return stats
    
    try:
        # Check log file age
        log_mtime = log_file.stat().st_mtime
        stats['last_log_time'] = datetime.fromtimestamp(log_mtime)
        
        # Consider log stale if older than 30 minutes
        time_since_update = datetime.now() - stats['last_log_time']
        stats['is_stale'] = time_since_update.total_seconds() > 1800  # 30 min
        
        with open(log_file) as f:
            lines = f.readlines()
        
        # Process last 200 lines
        recent_lines = lines[-200:]
        
        for line in recent_lines:
            if 'Teaching:' in line or 'Taught:' in line or 'lesson' in line.lower():
                stats['lessons_taught'] += 1
            elif 'Question:' in line or 'Asking:' in line:
                stats['questions_asked'] += 1
                # Extract conversation
                q = line.split(':')[1].strip() if ':' in line else line.strip()
                stats['recent_conversations'].append({
                    'q': q[:60], 
                    'a': 'Learning in progress...', 
                    'time': datetime.now().strftime('%H:%M')
                })
            elif 'Session Complete' in line:
                # Extract final vocabulary count from session summary
                try:
                    import re
                    # Look for pattern like "Vocabulary: 3584 → 3590 (+6)"
                    match = re.search(r'Vocabulary:\s*\d+\s*→\s*(\d+)', line)
                    if match:
                        stats['concepts_explained'] = int(match.group(1))
                except:
                    pass
        
        # Limit conversations to last 3
        stats['recent_conversations'] = stats['recent_conversations'][-3:]
        
    except Exception as e:
        print(f"Error reading teacher log: {e}")
    
    return stats

def get_atlas_stats():
    """Get current Atlas learning stats - reads directly from shared_brain.pkl"""
    stats = {
        'vocabulary': 0,
        'total_tokens': 0,
        'unique_contexts': 0,
        'teacher_lessons': 0,
        'questions_asked': 0,
        'concepts_explained': 0,
        'conversations': [],
        'improvements': [],
        'commits': [],
        'active_agents': 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'brain_file_age': None
    }
    
    # Load from shared brain pickle (the actual source of truth)
    brain_path = ATLAS_DIR / 'shared_brain.pkl'
    if brain_path.exists():
        try:
            # Get file age for debugging
            mtime = brain_path.stat().st_mtime
            stats['brain_file_age'] = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            
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
    
    # Get teacher stats from log
    teacher_stats = get_teacher_stats()
    stats['teacher_lessons'] = teacher_stats['lessons_taught']
    stats['questions_asked'] = teacher_stats['questions_asked']
    stats['concepts_explained'] = teacher_stats['concepts_explained']
    stats['conversations'] = teacher_stats['recent_conversations']
    stats['teacher_log_stale'] = teacher_stats['is_stale']
    stats['teacher_log_last_update'] = teacher_stats['last_log_time'].strftime('%H:%M') if teacher_stats['last_log_time'] else 'unknown'
    
    # Get improvements
    stats['improvements'] = get_improvements()
    
    # Get commits
    stats['commits'] = get_git_commits()[:3]
    
    # Get active agents
    stats['active_agents'] = get_active_agents()
    
    return stats

def format_report(stats):
    """Format detailed stats for Telegram"""
    report = f"""🧠 *Atlas Learning Update*

📊 *Brain Stats (Live from shared_brain.pkl):*
• Vocabulary: *{stats['vocabulary']:,}* words
• Total tokens: {stats['total_tokens']:,}
• Unique contexts: {stats['unique_contexts']:,}
• Brain updated: {stats.get('brain_file_age', 'unknown')}

📚 *Current Session Activity:*
• Lessons taught: {stats['teacher_lessons']}
• Questions asked: {stats['questions_asked']}
• Concepts explained: {stats['concepts_explained']}
• Active agents: {stats['active_agents']}
"""
    
    # Only show conversations if we have real, non-stale data
    if stats['conversations'] and not stats.get('teacher_log_stale', True):
        report += f"\n💬 *Recent Conversations (Current Session):*\n"
        for i, conv in enumerate(stats['conversations'], 1):
            report += f"\n{i}. 🕐 {conv.get('time', '??:??')}\n"
            q = conv['q'][:40] + '...' if len(conv['q']) > 40 else conv['q']
            a = conv['a'][:45] + '...' if len(conv['a']) > 45 else conv['a']
            report += f"   Q: _{q}_\n"
            report += f"   A: `{a}`\n"
    else:
        # Log is stale or no conversations - show status
        if stats.get('teacher_log_stale', True):
            last_update = stats.get('teacher_log_last_update', 'unknown')
            report += f"\n💬 *Recent Conversations:*\n"
            report += f"   _No active session. Last teacher activity: {last_update}_\n"
        else:
            report += "\n💬 *Recent Conversations:*\n   No conversations in current session yet.\n"
    
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
    print(f"Atlas Telegram Reporter - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    stats = get_atlas_stats()
    report = format_report(stats)
    
    print(f"\nStats collected:")
    print(f"  - Vocabulary: {stats['vocabulary']:,} words")
    print(f"  - Total tokens: {stats['total_tokens']:,}")
    print(f"  - Unique contexts: {stats['unique_contexts']:,}")
    print(f"  - Brain file age: {stats.get('brain_file_age', 'unknown')}")
    print(f"  - Teacher lessons: {stats['teacher_lessons']}")
    print(f"  - Active agents: {stats['active_agents']}")
    
    print(f"\nSending report...")
    success = send_telegram_message(report)
    
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}\n")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
