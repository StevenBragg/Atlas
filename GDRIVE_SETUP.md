# Google Drive Backup Setup for Atlas

## Quick Setup

1. **Go to Google Cloud Console:**
   - https://console.cloud.google.com/apis/credentials

2. **Enable Google Drive API:**
   - APIs & Services → Library
   - Search "Google Drive API"
   - Click Enable

3. **Create OAuth 2.0 Credentials:**
   - APIs & Services → Credentials
   - Click "Create Credentials" → "OAuth 2.0 Client ID"
   - Application type: "Desktop app"
   - Name: "Atlas Backup"
   - Click Create

4. **Download credentials:**
   - Click the download icon (⬇️) next to your new credential
   - Rename the file to `credentials.json`
   - Upload it to: `/root/.openclaw/workspace/Atlas/credentials/`

5. **Run first backup:**
   ```bash
   cd /root/.openclaw/workspace/Atlas
   source venv/bin/activate
   python3 gdrive_backup.py
   ```

6. **Authenticate:**
   - The script will provide a URL
   - Open it in your browser
   - Sign in with your Google account
   - Allow access to Google Drive
   - Copy the authorization code back to the terminal

7. **Done!** Future backups will use the saved token.

## Automatic Backups

To backup every hour, add to crontab:
```bash
0 * * * * cd /root/.openclaw/workspace/Atlas && source venv/bin/activate && python3 gdrive_backup.py
```

Or run manually whenever you want:
```bash
python3 gdrive_backup.py
```

## Files Backed Up

- `teacher_state/` - Atlas's learned weights and memory
- `improvements/state/` - Autonomous improvement state
- `checkpoints/` - Full system checkpoints

All files are timestamped so you can restore from any point in time.
