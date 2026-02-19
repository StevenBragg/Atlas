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
        result = subprocess.run(['pgrep', '-f', 'autonomous'], 
                              capture_output=True, text=True)
        return len([p for p in result.stdout.strip().split('\n') if p]) if result.stdout else 0
    except:
        return 0

def get_teacher_stats():
    """Get stats from teacher log - only most recent session"""
    teacher_log = ATLAS_DIR / 'logs/teacher_agent.log'
    stats = {
        'lessons_taught': 0,
        'questions_asked': 0,
        'concepts_explained': 0,
        'recent_conversations': []
    }
    
    if not teacher_log.exists():
        return stats
    
    try:
        with open(teacher_log) as f:
            lines = f.readlines()
        
        # Find the last restart (most recent session)
        last_restart_idx = 0
        for i, line in enumerate(lines):
            if '📚 Initializing Autonomous Teacher' in line:
                last_restart_idx = i
        
        # Only process lines from the most recent session
        recent_lines = lines[last_restart_idx:]
        
        for line in recent_lines:
            if '📖 Teaching lesson on:' in line:
                stats['lessons_taught'] += 1
            elif '❓ Asking:' in line:
                stats['questions_asked'] += 1
                q = line.split('❓ Asking:')[1].strip()
                stats['recent_conversations'].append({
                    'q': q, 
                    'a': '...', 
                    'time': datetime.now().strftime('%H:%M')
                })
            elif 'Tokens learned:' in line:
                try:
                    tokens = int(line.split('Tokens learned:')[1].strip())
                    stats['concepts_explained'] += tokens
                except:
                    pass
            elif 'Atlas answered:' in line and stats['recent_conversations']:
                a = line.split('Atlas answered:')[1].strip()
                stats['recent_conversations'][-1]['a'] = a[:80]
        
        # Keep only last 3 conversations
        stats['recent_conversations'] = stats['recent_conversations'][-3:]
        
    except Exception as e:
        print(f"Teacher log error: {e}")
    
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
    
    # Only show conversations if we have recent ones
    if stats['conversations']:
        report += f"\n💬 *Recent Conversations (Current Session):*\n"
        for i, conv in enumerate(stats['conversations'], 1):
            report += f"\n{i}. 🕐 {conv.get('time', '??:??')}\n"
            q = conv['q'][:40] + '...' if len(conv['q']) > 40 else conv['q']
            a = conv['a'][:45] + '...' if len(conv['a']) > 45 else conv['a']
            report += f"   Q: _{q}_\n"
            report += f"   A: `{a}`\n"
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
