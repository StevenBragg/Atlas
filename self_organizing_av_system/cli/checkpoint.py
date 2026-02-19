#!/usr/bin/env python3
"""
Checkpoint Management CLI for Atlas System

Provides command-line interface for:
- Creating checkpoints
- Listing checkpoints
- Restoring checkpoints
- Rolling back to previous versions
- Managing cloud sync
- Configuring auto-checkpoint settings

Usage:
    python -m self_organizing_av_system.cli.checkpoint [command] [options]
    
Or when installed:
    atlas-checkpoint [command] [options]
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir.parent.parent))

try:
    from self_organizing_av_system.core.checkpoint_manager import (
        CheckpointManager, CheckpointInfo, get_checkpoint_manager
    )
    from self_organizing_av_system.config.configuration import SystemConfig
except ImportError as e:
    print(f"Error importing Atlas modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or environment."""
    config = {}
    
    # Try to load from file
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    # Override with environment variables
    env_mappings = {
        'ATLAS_CHECKPOINT_DIR': 'checkpoint_dir',
        'ATLAS_CHECKPOINT_INTERVAL': 'checkpoint_interval_minutes',
        'ATLAS_MAX_CHECKPOINTS': 'max_checkpoints',
        'ATLAS_S3_BUCKET': 's3_bucket',
        'ATLAS_S3_REGION': 's3_region',
        'ATLAS_S3_ENDPOINT': 's3_endpoint',
        'ATLAS_S3_ACCESS_KEY': 's3_access_key',
        'ATLAS_S3_SECRET_KEY': 's3_secret_key',
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            # Convert to appropriate type
            if config_key in ['checkpoint_interval_minutes', 'max_checkpoints']:
                value = int(value)
            config[config_key] = value
    
    return config


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def cmd_save(args):
    """Save a checkpoint."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    
    # Import system classes
    try:
        from self_organizing_av_system.core.system import SelfOrganizingAVSystem
        from self_organizing_av_system.models.visual.processor import VisualProcessor
        from self_organizing_av_system.models.audio.processor import AudioProcessor
    except ImportError:
        logger.error("Could not import system classes")
        return 1
    
    # Initialize system (or load from existing checkpoint)
    try:
        if args.from_system:
            # Load from a running system state
            logger.info(f"Loading from system state: {args.from_system}")
            # This would require pickle loading or similar
            with open(args.from_system, 'rb') as f:
                system = pickle.load(f)
        else:
            # Create new system instance
            logger.info("Initializing new system instance...")
            visual_processor = VisualProcessor()
            audio_processor = AudioProcessor()
            system = SelfOrganizingAVSystem(
                visual_processor=visual_processor,
                audio_processor=audio_processor
            )
            
            # Try to load latest checkpoint if requested
            if args.incremental:
                checkpoints = manager.list_checkpoints()
                if checkpoints:
                    latest = checkpoints[0]
                    logger.info(f"Loading incremental from v{latest['version']}...")
                    system, _ = manager.load_checkpoint(version=latest['version'])
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return 1
    
    # Create checkpoint
    metadata = {
        "command": "cli_save",
        "description": args.description or "CLI checkpoint",
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
        
        print(f"\n✓ Checkpoint saved successfully!")
        print(f"  Version:    {info.version}")
        print(f"  Name:       {info.name}")
        print(f"  Size:       {format_size(info.size_bytes)}")
        print(f"  Compressed: {'Yes' if info.compressed else 'No'}")
        print(f"  Cloud:      {'Yes' if info.cloud_synced else 'No'}")
        print(f"  Path:       {info.path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return 1


def cmd_load(args):
    """Load a checkpoint."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
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
        
        print(f"\n✓ Checkpoint loaded successfully!")
        print(f"  Version: {info.version}")
        print(f"  Name:    {info.name}")
        print(f"  Created: {info.created_at}")
        
        # Save to output file if specified
        if args.output:
            logger.info(f"Saving to {args.output}...")
            if hasattr(system, 'save_state'):
                system.save_state(args.output)
            else:
                import pickle
                with open(args.output, 'wb') as f:
                    pickle.dump(system, f)
            print(f"  Saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 1


def cmd_list(args):
    """List checkpoints."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    checkpoints = manager.list_checkpoints(include_metadata=args.verbose)
    
    if not checkpoints:
        print("No checkpoints found.")
        return 0
    
    # Filter by tag if specified
    if args.tag:
        checkpoints = [
            cp for cp in checkpoints 
            if args.tag in (cp.get('metadata', {}).get('tags', []))
        ]
    
    if args.json:
        print(json.dumps(checkpoints, indent=2))
        return 0
    
    print(f"\n{'Ver':<5} {'Name':<28} {'Size':<12} {'Cmp':<4} {'Cloud':<6} {'Created'}")
    print("-" * 90)
    
    for cp in checkpoints:
        created = cp['created_at'][:19] if len(cp['created_at']) > 19 else cp['created_at']
        print(f"{cp['version']:<5} {cp['name']:<28} {format_size(cp['size_bytes']):<12} "
              f"{'Y' if cp['compressed'] else 'N':<4} "
              f"{'Y' if cp['cloud_synced'] else 'N':<6} {created}")
        
        if args.verbose and cp.get('metadata'):
            meta = cp['metadata']
            if meta.get('description'):
                print(f"      Description: {meta['description']}")
            if meta.get('tags'):
                print(f"      Tags: {', '.join(meta['tags'])}")
    
    # Summary
    stats = manager.get_stats()
    print(f"\nTotal: {stats['total_checkpoints']} checkpoints, "
          f"{format_size(stats['total_size_bytes'])}")
    
    return 0


def cmd_rollback(args):
    """Rollback to a previous version."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    
    try:
        info = manager.rollback(steps=args.steps)
        
        print(f"\n✓ Rolled back {args.steps} version(s)")
        print(f"  Now at version {info.version}: {info.name}")
        print(f"  Created: {info.created_at}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error rolling back: {e}")
        return 1


def cmd_delete(args):
    """Delete a checkpoint."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    
    # Get confirmation
    if not args.yes:
        if args.version:
            target = f"version {args.version}"
        elif args.name:
            target = f"'{args.name}'"
        else:
            logger.error("Must specify --version or --name")
            return 1
        
        confirm = input(f"Delete checkpoint {target}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0
    
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
                logger.error(f"Checkpoint not found: {args.name}")
                return 1
            success = manager.delete_checkpoint(version)
        else:
            logger.error("Must specify --version or --name")
            return 1
        
        if success:
            print("\n✓ Checkpoint deleted successfully!")
            return 0
        else:
            logger.error("Failed to delete checkpoint")
            return 1
            
    except Exception as e:
        logger.error(f"Error deleting checkpoint: {e}")
        return 1


def cmd_sync(args):
    """Sync checkpoints with cloud storage."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    # Enable cloud sync
    config['cloud_sync'] = True
    
    manager = get_checkpoint_manager(config)
    
    if args.to_cloud:
        logger.info("Syncing to cloud...")
        synced = manager.sync_to_cloud(version=args.version)
        print(f"\n✓ Synced {len(synced)} checkpoint(s) to cloud")
        for info in synced:
            print(f"  v{info.version}: {info.name} ({format_size(info.size_bytes)})")
            
    elif args.from_cloud:
        logger.info("Syncing from cloud...")
        downloaded = manager.sync_from_cloud()
        print(f"\n✓ Downloaded {len(downloaded)} checkpoint(s) from cloud")
        for info in downloaded:
            print(f"  v{info.version}: {info.name} ({format_size(info.size_bytes)})")
    else:
        print("Use --to-cloud or --from-cloud to specify sync direction")
        return 1
    
    return 0


def cmd_auto(args):
    """Manage auto-checkpoint settings."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    
    if args.status:
        stats = manager.get_stats()
        print("\nAuto-Checkpoint Status")
        print("-" * 30)
        print(f"Running:     {'Yes' if stats['auto_checkpoint_running'] else 'No'}")
        print(f"Interval:    {config.get('checkpoint_interval_minutes', 10)} minutes")
        print(f"Directory:   {config.get('checkpoint_dir', 'checkpoints')}")
        print(f"Max kept:    {config.get('max_checkpoints', 10)}")
        return 0
    
    if args.start:
        try:
            from self_organizing_av_system.core.system import SelfOrganizingAVSystem
            from self_organizing_av_system.models.visual.processor import VisualProcessor
            from self_organizing_av_system.models.audio.processor import AudioProcessor
            
            visual_processor = VisualProcessor()
            audio_processor = AudioProcessor()
            system = SelfOrganizingAVSystem(
                visual_processor=visual_processor,
                audio_processor=audio_processor
            )
            
            interval = args.interval or config.get('checkpoint_interval_minutes', 10)
            manager.start_auto_checkpoint(system, interval_minutes=interval)
            
            print(f"\n✓ Auto-checkpoint started (interval: {interval} minutes)")
            print("  Press Ctrl+C to stop")
            
            # Keep running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_auto_checkpoint()
                print("\n✓ Auto-checkpoint stopped")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error starting auto-checkpoint: {e}")
            return 1
    
    if args.stop:
        manager.stop_auto_checkpoint()
        print("\n✓ Auto-checkpoint stopped")
        return 0
    
    return 0


def cmd_verify(args):
    """Verify checkpoint integrity."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    
    try:
        if args.all:
            print("Verifying all checkpoints...")
            checkpoints = manager.list_checkpoints(include_metadata=True)
            all_valid = True
            
            for cp in checkpoints:
                info = manager.get_checkpoint_info(cp['version'])
                if not info or not info.checksum:
                    print(f"  ⚠ v{cp['version']}: No checksum")
                    continue
                
                path = Path(info.path)
                if not path.exists():
                    print(f"  ✗ v{cp['version']}: File not found")
                    all_valid = False
                    continue
                
                current_checksum = manager._compute_checksum(path)
                if current_checksum == info.checksum:
                    print(f"  ✓ v{cp['version']}: Valid")
                else:
                    print(f"  ✗ v{cp['version']}: Checksum mismatch!")
                    all_valid = False
            
            return 0 if all_valid else 1
        
        elif args.version:
            info = manager.get_checkpoint_info(args.version)
            if not info:
                logger.error(f"Checkpoint v{args.version} not found")
                return 1
            
            if not info.checksum:
                print(f"No checksum available for v{args.version}")
                return 1
            
            path = Path(info.path)
            if not path.exists():
                logger.error(f"File not found: {path}")
                return 1
            
            current_checksum = manager._compute_checksum(path)
            
            if current_checksum == info.checksum:
                print(f"\n✓ Checkpoint v{args.version} is valid")
                print(f"  Checksum: {info.checksum}")
                return 0
            else:
                print(f"\n✗ Checkpoint v{args.version} is corrupted!")
                print(f"  Expected: {info.checksum}")
                print(f"  Actual:   {current_checksum}")
                return 1
        else:
            logger.error("Must specify --version or --all")
            return 1
            
    except Exception as e:
        logger.error(f"Error verifying checkpoint: {e}")
        return 1


def cmd_stats(args):
    """Show checkpoint statistics."""
    config = load_config(args.config)
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    manager = get_checkpoint_manager(config)
    stats = manager.get_stats()
    
    print("\nCheckpoint Statistics")
    print("-" * 40)
    print(f"Total checkpoints:  {stats['total_checkpoints']}")
    print(f"Current version:    {stats['current_version']}")
    print(f"Total size:         {format_size(stats['total_size_bytes'])}")
    print(f"Cloud synced:       {stats['cloud_synced']}")
    print(f"Compressed:         {stats['compressed']}")
    print(f"Auto-checkpoint:    {'Running' if stats['auto_checkpoint_running'] else 'Stopped'}")
    print(f"Cloud available:    {'Yes' if stats['cloud_available'] else 'No'}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Atlas Checkpoint Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s save --name my_checkpoint --compress
  %(prog)s load --version 42
  %(prog)s list --verbose
  %(prog)s rollback --steps 1
  %(prog)s sync --to-cloud
  %(prog)s auto --start --interval 5
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Configuration file (JSON or YAML)'
    )
    parser.add_argument(
        '-d', '--checkpoint-dir',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save a checkpoint')
    save_parser.add_argument('-n', '--name', help='Checkpoint name')
    save_parser.add_argument('-d', '--description', help='Description')
    save_parser.add_argument('--compress', action='store_true', help='Compress checkpoint')
    save_parser.add_argument('--cloud', action='store_true', help='Sync to cloud')
    save_parser.add_argument('--from-system', help='Load from system state file')
    save_parser.add_argument('--incremental', action='store_true', help='Start from latest')
    save_parser.add_argument('--tags', help='Comma-separated tags')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a checkpoint')
    load_parser.add_argument('--version', type=int, help='Version number')
    load_parser.add_argument('--name', help='Checkpoint name')
    load_parser.add_argument('-o', '--output', help='Output file for loaded system')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List checkpoints')
    list_parser.add_argument('--json', action='store_true', help='Output as JSON')
    list_parser.add_argument('--tag', help='Filter by tag')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous version')
    rollback_parser.add_argument('--steps', type=int, default=1, help='Number of versions to rollback')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a checkpoint')
    delete_parser.add_argument('--version', type=int, help='Version number')
    delete_parser.add_argument('--name', help='Checkpoint name')
    delete_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync with cloud storage')
    sync_parser.add_argument('--to-cloud', action='store_true', help='Upload to cloud')
    sync_parser.add_argument('--from-cloud', action='store_true', help='Download from cloud')
    sync_parser.add_argument('--version', type=int, help='Specific version to sync')
    
    # Auto command
    auto_parser = subparsers.add_parser('auto', help='Manage auto-checkpoint')
    auto_parser.add_argument('--start', action='store_true', help='Start auto-checkpoint')
    auto_parser.add_argument('--stop', action='store_true', help='Stop auto-checkpoint')
    auto_parser.add_argument('--status', action='store_true', help='Show status')
    auto_parser.add_argument('--interval', type=int, help='Interval in minutes')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify checkpoint integrity')
    verify_parser.add_argument('--version', type=int, help='Version to verify')
    verify_parser.add_argument('--all', action='store_true', help='Verify all checkpoints')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    commands = {
        'save': cmd_save,
        'load': cmd_load,
        'list': cmd_list,
        'rollback': cmd_rollback,
        'delete': cmd_delete,
        'sync': cmd_sync,
        'auto': cmd_auto,
        'verify': cmd_verify,
        'stats': cmd_stats,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
