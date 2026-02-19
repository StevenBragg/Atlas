#!/usr/bin/env python3
"""
Backup Script for Atlas System

Provides manual backup functionality with options for:
- Full system backup
- Incremental backups
- Cloud sync
- Compression control
- Retention management

Usage:
    python scripts/backup.py [command] [options]
    
Commands:
    create      Create a new backup
    list        List available backups
    restore     Restore from a backup
    delete      Delete a backup
    sync        Sync backups to/from cloud
    verify      Verify backup integrity
    
Examples:
    python scripts/backup.py create --name my_backup --compress
    python scripts/backup.py list --verbose
    python scripts/backup.py restore --version 42
    python scripts/backup.py sync --to-cloud
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

try:
    from self_organizing_av_system.core.checkpoint_manager import (
        CheckpointManager, CheckpointInfo, get_checkpoint_manager
    )
    from self_organizing_av_system.core.system import SelfOrganizingAVSystem
    from self_organizing_av_system.models.visual.processor import VisualProcessor
    from self_organizing_av_system.models.audio.processor import AudioProcessor
except ImportError as e:
    print(f"Error importing Atlas modules: {e}")
    print("Make sure you're running from the Atlas directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_system(config_path: Optional[str] = None) -> Optional[SelfOrganizingAVSystem]:
    """Initialize or load the Atlas system."""
    try:
        # Default configuration
        visual_config = {
            "input_width": 64,
            "input_height": 64,
            "use_grayscale": True,
            "patch_size": 8,
            "stride": 4,
            "layer_sizes": [200, 100, 50]
        }
        
        audio_config = {
            "sample_rate": 22050,
            "n_mels": 64,
            "layer_sizes": [150, 75, 40]
        }
        
        system_config = {
            "multimodal_size": 100,
            "learning_rate": 0.01
        }
        
        # Initialize processors
        visual_processor = VisualProcessor(**visual_config)
        audio_processor = AudioProcessor(**audio_config)
        
        # Initialize system
        system = SelfOrganizingAVSystem(
            visual_processor=visual_processor,
            audio_processor=audio_processor,
            config=system_config
        )
        
        return system
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return None


def cmd_create(args):
    """Create a new backup."""
    logger.info("Creating backup...")
    
    # Get checkpoint manager
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
        "compression": args.compress,
        "cloud_sync": args.cloud,
        "s3_bucket": args.s3_bucket,
        "s3_region": args.s3_region,
        "s3_endpoint": args.s3_endpoint,
        "s3_access_key": args.s3_access_key,
        "s3_secret_key": args.s3_secret_key,
    }
    
    manager = get_checkpoint_manager(config)
    
    # Get or create system
    system = get_system(args.config)
    if not system:
        logger.error("Failed to initialize system")
        return 1
    
    # Try to load latest checkpoint if requested
    if args.from_latest:
        try:
            checkpoints = manager.list_checkpoints()
            if checkpoints:
                latest_version = checkpoints[0]["version"]
                logger.info(f"Loading latest checkpoint (v{latest_version})...")
                system, _ = manager.load_checkpoint(version=latest_version)
        except Exception as e:
            logger.warning(f"Could not load latest checkpoint: {e}")
    
    # Create backup
    metadata = {
        "source": "manual_backup",
        "description": args.description or "Manual backup",
        "created_by": args.user or "unknown",
        "tags": args.tags.split(",") if args.tags else [],
    }
    
    try:
        info = manager.save_checkpoint(
            system=system,
            name=args.name,
            metadata=metadata,
            compress=args.compress,
            sync_to_cloud=args.cloud
        )
        
        print(f"\n✓ Backup created successfully!")
        print(f"  Name: {info.name}")
        print(f"  Version: {info.version}")
        print(f"  Size: {info.size_bytes:,} bytes")
        print(f"  Compressed: {info.compressed}")
        print(f"  Cloud synced: {info.cloud_synced}")
        print(f"  Location: {info.path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return 1


def cmd_list(args):
    """List available backups."""
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
    }
    
    manager = get_checkpoint_manager(config)
    checkpoints = manager.list_checkpoints(include_metadata=args.verbose)
    
    if not checkpoints:
        print("No backups found.")
        return 0
    
    print(f"\n{'Version':<8} {'Name':<30} {'Size':<12} {'Compressed':<10} {'Cloud':<6} {'Created'}")
    print("-" * 100)
    
    for cp in checkpoints:
        size_str = f"{cp['size_bytes']:,} B"
        if cp['size_bytes'] > 1024 * 1024:
            size_str = f"{cp['size_bytes'] / (1024 * 1024):.2f} MB"
        elif cp['size_bytes'] > 1024:
            size_str = f"{cp['size_bytes'] / 1024:.2f} KB"
        
        print(f"{cp['version']:<8} {cp['name']:<30} {size_str:<12} "
              f"{'Yes' if cp['compressed'] else 'No':<10} "
              f"{'Yes' if cp['cloud_synced'] else 'No':<6} "
              f"{cp['created_at'][:19]}")
        
        if args.verbose and cp.get('metadata'):
            print(f"  Metadata: {cp['metadata']}")
            print(f"  Checksum: {cp.get('checksum', 'N/A')}")
    
    # Print stats
    stats = manager.get_stats()
    print(f"\nTotal: {stats['total_checkpoints']} backups, "
          f"{stats['total_size_bytes'] / (1024 * 1024):.2f} MB")
    
    return 0


def cmd_restore(args):
    """Restore from a backup."""
    logger.info("Restoring backup...")
    
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
    }
    
    manager = get_checkpoint_manager(config)
    
    try:
        if args.version:
            system, info = manager.load_checkpoint(version=args.version)
        elif args.name:
            system, info = manager.load_checkpoint(name=args.name)
        else:
            # Load latest
            checkpoints = manager.list_checkpoints()
            if not checkpoints:
                logger.error("No checkpoints available")
                return 1
            latest_version = checkpoints[0]["version"]
            logger.info(f"Loading latest checkpoint (v{latest_version})...")
            system, info = manager.load_checkpoint(version=latest_version)
        
        print(f"\n✓ Backup restored successfully!")
        print(f"  Name: {info.name}")
        print(f"  Version: {info.version}")
        print(f"  Created: {info.created_at}")
        
        # Save restored state as current if requested
        if args.save_as_current:
            current_path = Path(config["checkpoint_dir"]) / "current_state.pkl"
            if hasattr(system, 'save_state'):
                system.save_state(str(current_path))
                print(f"  Saved as current state: {current_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return 1


def cmd_delete(args):
    """Delete a backup."""
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
    }
    
    manager = get_checkpoint_manager(config)
    
    try:
        if args.version:
            success = manager.delete_checkpoint(args.version)
        elif args.name:
            # Find version by name
            checkpoints = manager.list_checkpoints(include_metadata=True)
            version = None
            for cp in checkpoints:
                if cp['name'] == args.name:
                    version = cp['version']
                    break
            if version is None:
                logger.error(f"Backup not found: {args.name}")
                return 1
            success = manager.delete_checkpoint(version)
        else:
            logger.error("Must specify --version or --name")
            return 1
        
        if success:
            print(f"\n✓ Backup deleted successfully!")
            return 0
        else:
            logger.error("Failed to delete backup")
            return 1
            
    except Exception as e:
        logger.error(f"Error deleting backup: {e}")
        return 1


def cmd_sync(args):
    """Sync backups to/from cloud."""
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
        "cloud_sync": True,
        "s3_bucket": args.s3_bucket,
        "s3_region": args.s3_region,
        "s3_endpoint": args.s3_endpoint,
        "s3_access_key": args.s3_access_key,
        "s3_secret_key": args.s3_secret_key,
    }
    
    manager = get_checkpoint_manager(config)
    
    try:
        if args.to_cloud:
            logger.info("Syncing to cloud...")
            synced = manager.sync_to_cloud(version=args.version)
            print(f"\n✓ Synced {len(synced)} backup(s) to cloud")
            for info in synced:
                print(f"  - {info.name} ({info.size_bytes:,} bytes)")
                
        elif args.from_cloud:
            logger.info("Syncing from cloud...")
            downloaded = manager.sync_from_cloud()
            print(f"\n✓ Downloaded {len(downloaded)} backup(s) from cloud")
            for info in downloaded:
                print(f"  - {info.name} ({info.size_bytes:,} bytes)")
        else:
            print("Use --to-cloud or --from-cloud to specify sync direction")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error syncing: {e}")
        return 1


def cmd_verify(args):
    """Verify backup integrity."""
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
    }
    
    manager = get_checkpoint_manager(config)
    
    try:
        if args.version:
            info = manager.get_checkpoint_info(args.version)
        elif args.name:
            checkpoints = manager.list_checkpoints(include_metadata=True)
            info = None
            for cp in checkpoints:
                if cp['name'] == args.name:
                    info = manager.get_checkpoint_info(cp['version'])
                    break
        else:
            # Verify all
            checkpoints = manager.list_checkpoints(include_metadata=True)
            print(f"Verifying {len(checkpoints)} backup(s)...")
            all_valid = True
            for cp in checkpoints:
                info = manager.get_checkpoint_info(cp['version'])
                if not info or not info.checksum:
                    print(f"  ✗ v{cp['version']}: No checksum available")
                    all_valid = False
                    continue
                    
                from self_organizing_av_system.core.checkpoint_manager import CheckpointManager
                path = Path(info.path)
                if not path.exists():
                    print(f"  ✗ v{cp['version']}: File not found")
                    all_valid = False
                    continue
                
                current_checksum = CheckpointManager._compute_checksum(CheckpointManager, path)
                if current_checksum == info.checksum:
                    print(f"  ✓ v{cp['version']}: Valid")
                else:
                    print(f"  ✗ v{cp['version']}: Checksum mismatch!")
                    all_valid = False
            
            return 0 if all_valid else 1
        
        if not info:
            logger.error("Backup not found")
            return 1
        
        if not info.checksum:
            print("No checksum available for this backup")
            return 1
        
        path = Path(info.path)
        if not path.exists():
            logger.error(f"Backup file not found: {path}")
            return 1
        
        current_checksum = manager._compute_checksum(path)
        
        if current_checksum == info.checksum:
            print(f"\n✓ Backup integrity verified!")
            print(f"  Checksum: {info.checksum}")
            return 0
        else:
            print(f"\n✗ Checksum mismatch!")
            print(f"  Expected: {info.checksum}")
            print(f"  Actual:   {current_checksum}")
            return 1
            
    except Exception as e:
        logger.error(f"Error verifying backup: {e}")
        return 1


def cmd_stats(args):
    """Show backup statistics."""
    config = {
        "checkpoint_dir": args.checkpoint_dir or "checkpoints",
    }
    
    manager = get_checkpoint_manager(config)
    stats = manager.get_stats()
    
    print("\nBackup Statistics")
    print("-" * 40)
    print(f"Total backups:     {stats['total_checkpoints']}")
    print(f"Current version:   {stats['current_version']}")
    print(f"Total size:        {stats['total_size_bytes'] / (1024 * 1024):.2f} MB")
    print(f"Cloud synced:      {stats['cloud_synced']}")
    print(f"Compressed:        {stats['compressed']}")
    print(f"Auto-checkpoint:   {'Running' if stats['auto_checkpoint_running'] else 'Stopped'}")
    print(f"Cloud available:   {'Yes' if stats['cloud_available'] else 'No'}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Atlas System Backup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create --name my_backup --compress
  %(prog)s list --verbose
  %(prog)s restore --version 42
  %(prog)s sync --to-cloud
        """
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        help='Directory for checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--s3-bucket',
        help='S3 bucket for cloud sync'
    )
    parser.add_argument(
        '--s3-region',
        default='us-east-1',
        help='S3 region (default: us-east-1)'
    )
    parser.add_argument(
        '--s3-endpoint',
        help='S3 endpoint URL (for MinIO compatibility)'
    )
    parser.add_argument(
        '--s3-access-key',
        help='S3 access key'
    )
    parser.add_argument(
        '--s3-secret-key',
        help='S3 secret key'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new backup')
    create_parser.add_argument('--name', help='Backup name')
    create_parser.add_argument('--description', help='Backup description')
    create_parser.add_argument('--compress', action='store_true', help='Compress the backup')
    create_parser.add_argument('--cloud', action='store_true', help='Sync to cloud after creation')
    create_parser.add_argument('--from-latest', action='store_true', help='Start from latest checkpoint')
    create_parser.add_argument('--config', help='Path to system config file')
    create_parser.add_argument('--user', help='Username for metadata')
    create_parser.add_argument('--tags', help='Comma-separated tags')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available backups')
    list_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from a backup')
    restore_parser.add_argument('--version', type=int, help='Version number to restore')
    restore_parser.add_argument('--name', help='Backup name to restore')
    restore_parser.add_argument('--save-as-current', action='store_true', help='Save restored state as current')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a backup')
    delete_parser.add_argument('--version', type=int, help='Version number to delete')
    delete_parser.add_argument('--name', help='Backup name to delete')
    delete_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync backups to/from cloud')
    sync_parser.add_argument('--to-cloud', action='store_true', help='Sync to cloud')
    sync_parser.add_argument('--from-cloud', action='store_true', help='Sync from cloud')
    sync_parser.add_argument('--version', type=int, help='Specific version to sync')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('--version', type=int, help='Version number to verify')
    verify_parser.add_argument('--name', help='Backup name to verify')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show backup statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    commands = {
        'create': cmd_create,
        'list': cmd_list,
        'restore': cmd_restore,
        'delete': cmd_delete,
        'sync': cmd_sync,
        'verify': cmd_verify,
        'stats': cmd_stats,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
