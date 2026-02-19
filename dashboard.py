#!/usr/bin/env python3
"""
Atlas Learning Dashboard

Real-time web interface showing Atlas's learning progress,
conversations, and metrics.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify

sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')

app = Flask(__name__)

# Global state cache
dashboard_state = {
    'last_update': None,
    'teacher_stats': {},
    'autonomous_stats': {},
    'conversations': [],
    'vocabulary_size': 0,
    'learning_cycles': 0,
    'recent_improvements': []
}

def read_log_file(filepath, lines=50):
    """Read last N lines from log file"""
    try:
        with open(filepath, 'r') as f:
            all_lines = f.readlines()
            return all_lines[-lines:]
    except:
        return []

def parse_teacher_stats():
    """Parse teacher agent stats"""
    log_lines = read_log_file('/root/.openclaw/workspace/Atlas/logs/teacher_agent.log', 100)
    
    stats = {
        'lessons_taught': 0,
        'questions_asked': 0,
        'vocabulary': 0,
        'last_lesson': None,
        'last_qa': None,
        'status': 'Unknown'
    }
    
    conversations = []
    
    for line in log_lines:
        if 'Teaching lesson on:' in line:
            lesson = line.split('Teaching lesson on:')[1].strip()
            stats['last_lesson'] = lesson
            stats['lessons_taught'] += 1
        elif 'Asking:' in line:
            question = line.split('Asking:')[1].strip()
            stats['last_qa'] = {'question': question}
        elif 'Atlas answered:' in line:
            answer = line.split('Atlas answered:')[1].strip()
            if stats['last_qa']:
                stats['last_qa']['answer'] = answer
                conversations.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'question': stats['last_qa']['question'],
                    'answer': answer
                })
                stats['questions_asked'] += 1
        elif 'Vocabulary:' in line:
            try:
                vocab = int(line.split('Vocabulary:')[1].split()[0])
                stats['vocabulary'] = vocab
            except:
                pass
        elif 'Teaching Cycle' in line and 'complete' in line:
            stats['status'] = 'Active'
    
    return stats, conversations[-10:]  # Last 10 conversations

def parse_autonomous_stats():
    """Parse autonomous Atlas stats"""
    log_lines = read_log_file('/root/.openclaw/workspace/Atlas/logs/autonomous.out', 100)
    
    stats = {
        'files_read': 0,
        'lines_learned': 0,
        'ideas_generated': 0,
        'vocabulary': 0,
        'last_files': [],
        'status': 'Unknown'
    }
    
    for line in log_lines:
        if 'Learned from' in line:
            filename = line.split('Learned from')[1].split(':')[0].strip()
            stats['last_files'].append(filename)
            stats['files_read'] += 1
        elif 'Vocabulary:' in line:
            try:
                vocab = int(line.split('Vocabulary:')[1].split()[0])
                stats['vocabulary'] = vocab
            except:
                pass
        elif 'Learning cycle complete' in line:
            stats['status'] = 'Active'
        elif 'ideas generated:' in line.lower():
            try:
                ideas = int(line.split(':')[1].strip())
                stats['ideas_generated'] = ideas
            except:
                pass
    
    stats['last_files'] = stats['last_files'][-5:]  # Last 5 files
    return stats

def get_improvements():
    """Get recent improvement files"""
    improvements_dir = Path('/root/.openclaw/workspace/Atlas/improvements')
    if not improvements_dir.exists():
        return []
    
    files = sorted(improvements_dir.glob('improvement_*.py'), 
                   key=lambda p: p.stat().st_mtime, 
                   reverse=True)[:5]
    
    return [{'name': f.name, 'time': datetime.fromtimestamp(f.stat().st_mtime).strftime('%H:%M:%S')} 
            for f in files]

def update_state():
    """Update dashboard state"""
    global dashboard_state
    
    teacher_stats, conversations = parse_teacher_stats()
    autonomous_stats = parse_autonomous_stats()
    
    # Calculate total vocabulary (shared brain)
    vocab_size = max(teacher_stats.get('vocabulary', 0), 
                     autonomous_stats.get('vocabulary', 0))
    
    dashboard_state = {
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'teacher_stats': teacher_stats,
        'autonomous_stats': autonomous_stats,
        'conversations': conversations,
        'vocabulary_size': vocab_size,
        'learning_cycles': teacher_stats.get('lessons_taught', 0) + autonomous_stats.get('files_read', 0),
        'recent_improvements': get_improvements()
    }

def background_updater():
    """Background thread to update state every 5 seconds"""
    while True:
        update_state()
        time.sleep(5)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/state')
def api_state():
    """API endpoint for current state"""
    return jsonify(dashboard_state)

@app.route('/api/conversations')
def api_conversations():
    """API endpoint for conversations"""
    return jsonify(dashboard_state.get('conversations', []))

@app.route('/api/logs/<agent>')
def api_logs(agent):
    """API endpoint for raw logs"""
    if agent == 'teacher':
        logs = read_log_file('/root/.openclaw/workspace/Atlas/logs/teacher_agent.log', 30)
    elif agent == 'autonomous':
        logs = read_log_file('/root/.openclaw/workspace/Atlas/logs/autonomous.out', 30)
    else:
        logs = []
    
    return jsonify({'logs': logs})

if __name__ == '__main__':
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
