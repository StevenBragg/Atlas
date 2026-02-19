#!/usr/bin/env python3
"""
Atlas Telegram Reporter - Enhanced

Sends detailed regular updates about Atlas's learning progress to Telegram.
"""

import os
import sys
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

def get_git_commits():
    """Get recent git commits"""
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '-5'],
            capture_output=True,
            text=True,
            cwd='/root/.openclaw/workspace/Atlas'
        )
        return result.stdout.strip().split('\n')
    except:
        return []

def get_improvements():
    """Get recent improvement files"""
    imp_dir = Path('/root/.openclaw/workspace/Atlas/improvements')
    if not imp_dir.exists():
        return []
    
    files = sorted(imp_dir.glob('improvement_*.py'), 
                   key=lambda p: p.stat().st_mtime, 
                   reverse=True)[:3]
    return [f.name for f in files]

def get_active_agents():
    """Check running processes"""
    try:
        result = subprocess.run(['pgrep', '-f', 'autonomous'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip().split('\n')) if result.stdout else 0
    except:
        return 0

def get_atlas_stats():
    """Get current Atlas learning stats from the JSON file that teachers update"""
    import json
    
    stats = {
        'vocabulary': 0,
        'total_tokens': 0,
        'unique_contexts': 0,
        'teacher_lessons': 0,
        'files_read': 0,
        'conversations': [],
        'improvements': [],
        'commits': [],
        'active_agents': 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Try JSON brain first (what teachers actually update)
    json_paths = [
        Path('/root/.openclaw/workspace/Atlas/atlas_brain.json'),
        Path('/root/.openclaw/workspace/atlas_brain.json'),
    ]
    
    for json_path in json_paths:
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    stats['vocabulary'] = len(data.get('vocabulary', []))
                    stats['total_tokens'] = data.get('total_words_learned', 0)
                    stats['unique_contexts'] = len(data.get('word_pairs', []))
                    stats['teacher_lessons'] = data.get('exchanges', 0)
                    print(f"Loaded stats from {json_path}: {stats['vocabulary']} words")
                    break
            except Exception as e:
                print(f"Error loading JSON brain: {e}")
    
    # Fallback to pickle if JSON not found
    if stats['vocabulary'] == 0:
        brain_path = Path('/root/.openclaw/workspace/Atlas/shared_brain.pkl')
        if brain_path.exists():
            try:
                with open(brain_path, 'rb') as f:
                    state = pickle.load(f)
                    stats['vocabulary'] = len(state.get('token_to_idx', {}))
                    stats['total_tokens'] = state.get('total_tokens', 0)
                    stats['unique_contexts'] = len(state.get('context_weights', {}))
            except Exception as e:
                print(f"Error loading pickle brain: {e}")
    
    # Parse teacher logs
    teacher_log = Path('/root/.openclaw/workspace/Atlas/logs/teacher_agent.log')
    if teacher_log.exists():
        with open(teacher_log) as f:
            lines = f.readlines()
            for line in lines[-100:]:
                if 'Teaching lesson on:' in line:
                    stats['teacher_lessons'] += 1
                if 'Asking:' in line:
                    q = line.split('Asking:')[1].strip()
                    stats['conversations'].append({'q': q, 'a': '...', 'time': ''})
                if 'Atlas answered:' in line:
                    if stats['conversations']:
                        a = line.split('Atlas answered:')[1].strip()
                        stats['conversations'][-1]['a'] = a[:80]
                        stats['conversations'][-1]['time'] = datetime.now().strftime('%H:%M')
    
    # Parse autonomous logs
    auto_log = Path('/root/.openclaw/workspace/Atlas/logs/autonomous.out')
    if auto_log.exists():
        with open(auto_log) as f:
            content = f.read()
            stats['files_read'] = content.count('Learned from')
    
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

📊 *Brain Stats:*
• Vocabulary: *{stats['vocabulary']}* words
• Total tokens: {stats['total_tokens']:,}
• Unique contexts: {stats['unique_contexts']:,}

📚 *Learning Activity:*
• Lessons taught: {stats['teacher_lessons']}
• Files read: {stats['files_read']}
• Active agents: {stats['active_agents']}

💬 *Recent Conversations:*
"""
    
    for i, conv in enumerate(stats['conversations'][-3:], 1):
        report += f"\n{i}. 🕐 {conv.get('time', '??:??')}\n"
        report += f"   Q: _{conv['q'][:35]}..._\n"
        report += f"   A: `{conv['a'][:40]}...`\n"
    
    if stats['improvements']:
        report += f"\n🔧 *Recent Improvements:*\n"
        for imp in stats['improvements']:
            report += f"• {imp}\n"
    
    if stats['commits']:
        report += f"\n📦 *Recent Commits:*\n"
        for commit in stats['commits']:
            short = commit[:50] + '...' if len(commit) > 50 else commit
            report += f"• `{short}`\n"
    
    report += f"\n⏰ _{stats['timestamp']}_"
    
    return report

def send_telegram_message(message):
    """Send message via Telegram bot"""
    import urllib.request
    import urllib.parse
    
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("Telegram credentials not set")
        print(message)
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }).encode()
    
    try:
        with urllib.request.urlopen(url, data=data) as response:
            print("Message sent successfully")
    except Exception as e:
        print(f"Failed to send: {e}")
        print(message)

def main():
    stats = get_atlas_stats()
    report = format_report(stats)
    send_telegram_message(report)

if __name__ == '__main__':
    main()
