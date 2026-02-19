#!/usr/bin/env python3
"""
Atlas Team Monitor

Ensures sub-agents regularly commit their work.
Run this to check commit activity and remind agents to push.
"""

import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path

def check_git_commits():
    """Check recent commits"""
    result = subprocess.run(
        ['git', 'log', '--oneline', '-20', '--pretty=format:%h|%s|%ci'],
        capture_output=True,
        text=True,
        cwd='/root/.openclaw/workspace/Atlas'
    )
    
    commits = []
    for line in result.stdout.strip().split('\n'):
        if '|' in line:
            hash_, msg, date = line.split('|', 2)
            commits.append({
                'hash': hash_,
                'message': msg,
                'date': date
            })
    
    return commits

def check_commit_activity():
    """Check if there have been commits in the last hour"""
    commits = check_git_commits()
    
    if not commits:
        return False, "No commits found"
    
    latest = commits[0]
    latest_time = datetime.fromisoformat(latest['date'].replace('Z', '+00:00'))
    now = datetime.now(latest_time.tzinfo)
    
    hours_since = (now - latest_time).total_seconds() / 3600
    
    return hours_since < 1, f"Last commit: {hours_since:.1f} hours ago - {latest['message']}"

def main():
    print("=" * 60)
    print("🔍 Atlas Team Commit Monitor")
    print("=" * 60)
    print()
    
    # Check recent commits
    commits = check_git_commits()
    print(f"📊 Last {len(commits)} commits:")
    for c in commits[:5]:
        print(f"  {c['hash']}: {c['message'][:50]}")
    print()
    
    # Check activity
    active, status = check_commit_activity()
    print(f"⏰ Commit Activity: {status}")
    
    if active:
        print("✅ Team is committing regularly")
    else:
        print("⚠️  No recent commits - agents should push their work!")
    
    print()
    print("💡 Reminder: Agents should commit every 30-60 minutes")
    print("   Even incomplete work should be pushed to main branch")

if __name__ == "__main__":
    main()
