#!/usr/bin/env python3
"""
Google Drive Backup for Atlas

Automatically backs up Atlas checkpoints to Google Drive.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive folder ID
DRIVE_FOLDER_ID = "1owqo8QipS1I3QZYr7c5mM_cmTeQ9oYiG"

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']


class GoogleDriveBackup:
    """Backup Atlas state to Google Drive"""
    
    def __init__(self):
        self.service = None
        self.folder_id = DRIVE_FOLDER_ID
        self.local_base = Path('/root/.openclaw/workspace/Atlas')
        
    def authenticate(self):
        """Authenticate with Google Drive"""
        creds = None
        token_path = self.local_base / 'credentials' / 'token.pickle'
        
        # Load existing credentials
        if token_path.exists():
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, need to authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Need credentials.json file for initial setup
                creds_path = self.local_base / 'credentials' / 'credentials.json'
                if not creds_path.exists():
                    print("❌ No credentials.json found!")
                    print("Please download from Google Cloud Console:")
                    print("https://console.cloud.google.com/apis/credentials")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_path), SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            token_path.parent.mkdir(exist_ok=True)
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('drive', 'v3', credentials=creds)
        return True
    
    def upload_file(self, filepath, filename=None):
        """Upload a file to Google Drive"""
        if not self.service:
            if not self.authenticate():
                return None
        
        if filename is None:
            filename = filepath.name
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        drive_filename = f"{timestamp}_{filename}"
        
        file_metadata = {
            'name': drive_filename,
            'parents': [self.folder_id]
        }
        
        media = MediaFileUpload(str(filepath), resumable=True)
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name'
        ).execute()
        
        print(f"✅ Uploaded: {drive_filename} (ID: {file.get('id')})")
        return file.get('id')
    
    def backup_atlas_state(self):
        """Backup all Atlas state files"""
        print("☁️  Starting Google Drive backup...")
        
        if not self.authenticate():
            print("❌ Authentication failed")
            return []
        
        uploaded = []
        
        # Find all state files
        state_files = []
        
        # Teacher state
        teacher_state = self.local_base / 'teacher_state'
        if teacher_state.exists():
            state_files.extend(teacher_state.glob('*.pkl'))
            state_files.extend(teacher_state.glob('*.json'))
        
        # Improvements state
        improvements = self.local_base / 'improvements' / 'state'
        if improvements.exists():
            state_files.extend(improvements.glob('*.pkl'))
        
        # Checkpoints
        checkpoints = self.local_base / 'checkpoints'
        if checkpoints.exists():
            for cp_file in checkpoints.rglob('*.pkl'):
                state_files.append(cp_file)
        
        # Upload each file
        for filepath in state_files:
            try:
                file_id = self.upload_file(filepath)
                if file_id:
                    uploaded.append(file_id)
            except Exception as e:
                print(f"❌ Failed to upload {filepath}: {e}")
        
        print(f"\n✅ Backed up {len(uploaded)} files to Google Drive")
        print(f"📁 Folder: https://drive.google.com/drive/folders/{self.folder_id}")
        
        return uploaded


def main():
    backup = GoogleDriveBackup()
    backup.backup_atlas_state()


if __name__ == "__main__":
    main()
