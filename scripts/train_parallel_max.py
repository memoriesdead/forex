#!/usr/bin/env python3
"""
Parallel Training Script - Maximum RTX 5080 + Ryzen 9 9900X Utilization
========================================================================

Uses pipeline parallelism:
- CPU: 4 workers generate features simultaneously
- GPU: 3 workers train models concurrently

Expected speedup: ~3x (10 hours -> 3.5 hours for 52 pairs)

Usage:
    # Train all untrained pairs
    python scripts/train_parallel_max.py --all

    # Train specific tier
    python scripts/train_parallel_max.py --tier eur_crosses

    # Train specific pairs
    python scripts/train_parallel_max.py --symbols AUDCAD,AUDHKD,CADNOK,CADSEK

    # Dry run (show what would be trained)
    python scripts/train_parallel_max.py --all --dry-run

    # Monitor GPU utilization during training
    python scripts/train_parallel_max.py --all --monitor

    # Custom parallelism settings
    python scripts/train_parallel_max.py --all --cpu-workers 4 --gpu-workers 3

    # Clear feature cache (free disk space)
    python scripts/train_parallel_max.py --clear-cache
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Symbol tiers (same as train_all_mega.py)
SYMBOL_TIERS = {
    'majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
    'crosses': ['EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD'],
    'exotics': ['EURNZD', 'GBPNZD', 'AUDNZD', 'NZDJPY', 'AUDCAD', 'CADCHF', 'CADJPY'],
    'eur_crosses': ['EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK'],
    'gbp_crosses': ['GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNOK', 'GBPNZD', 'GBPSEK', 'GBPSGD'],
    'jpy_crosses': ['AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDJPY', 'SGDJPY', 'USDJPY'],
    'aud_crosses': ['AUDCAD', 'AUDCHF', 'AUDHKD', 'AUDJPY', 'AUDNZD', 'AUDSGD', 'AUDUSD'],
    'cad_crosses': ['AUDCAD', 'CADCHF', 'CADHKD', 'CADJPY', 'CADNOK', 'CADSEK', 'CADSGD', 'USDCAD'],
    'chf_crosses': ['AUDCHF', 'CADCHF', 'CHFJPY', 'CHFNOK', 'CHFSEK', 'EURCHF', 'GBPCHF', 'NZDCHF', 'USDCHF'],
    'nzd_crosses': ['AUDNZD', 'EURNZD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDHKD', 'NZDJPY', 'NZDSGD', 'NZDUSD'],
    'sgd_crosses': ['AUDSGD', 'CADSGD', 'CHFSGD', 'EURSGD', 'GBPSGD', 'NZDSGD', 'SGDJPY', 'USDSGD'],
    'scandi': ['EURNOK', 'EURSEK', 'GBPNOK', 'GBPSEK', 'CADNOK', 'CADSEK', 'CHFNOK', 'CHFSEK', 'NOKSEK', 'USDNOK', 'USDSEK'],
}


def show_status():
    """Show training status - what's trained and what's remaining."""
    from core.ml.parallel_trainer import ParallelTrainer

    trainer = ParallelTrainer()
    trained = trainer._get_already_trained()
    available = trainer._get_pairs_with_data()

    remaining = [p for p in available if p not in trained]

    print(f"\n{'='*60}")
    print("TRAINING STATUS")
    print(f"{'='*60}")
    print(f"Pairs with data: {len(available)}")
    print(f"Already trained: {len(trained)}")
    print(f"Remaining: {len(remaining)}")

    if trained:
        print(f"\nTrained: {', '.join(sorted(trained))}")

    if remaining:
        print(f"\nRemaining: {', '.join(sorted(remaining))}")

    # Show by tier
    print(f"\n{'='*60}")
    print("BY TIER:")
    print(f"{'='*60}")
    for tier, symbols in SYMBOL_TIERS.items():
        tier_trained = [s for s in symbols if s in trained]
        tier_remaining = [s for s in symbols if s in remaining]
        tier_nodata = [s for s in symbols if s not in available]

        print(f"\n{tier.upper()} ({len(symbols)} pairs):")
        print(f"  Trained: {len(tier_trained)}/{len(symbols)}")
        if tier_remaining:
            print(f"  Remaining: {', '.join(tier_remaining)}")
        if tier_nodata:
            print(f"  No data: {', '.join(tier_nodata)}")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel training with maximum GPU + CPU utilization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all untrained pairs
    python scripts/train_parallel_max.py --all

    # Train specific tier
    python scripts/train_parallel_max.py --tier eur_crosses

    # Train specific pairs
    python scripts/train_parallel_max.py --symbols AUDCAD,AUDHKD

    # Check status
    python scripts/train_parallel_max.py --status
        """
    )

    # Training targets
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='Train all pairs with data')
    group.add_argument('--tier', choices=list(SYMBOL_TIERS.keys()), help='Train specific tier')
    group.add_argument('--symbols', type=str, help='Comma-separated symbols to train')
    group.add_argument('--status', action='store_true', help='Show training status')
    group.add_argument('--clear-cache', action='store_true', help='Clear feature cache')

    # Parallelism settings
    parser.add_argument('--cpu-workers', type=int, default=4,
                       help='Number of CPU workers for feature generation (default: 4)')
    parser.add_argument('--gpu-workers', type=int, default=3,
                       help='Number of concurrent GPU training jobs (default: 3)')

    # Options
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Max training samples per pair (default: 50000)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Retrain already trained pairs')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be trained, but do not train')
    parser.add_argument('--monitor', action='store_true', default=True,
                       help='Print progress updates (default: True)')
    parser.add_argument('--no-monitor', action='store_false', dest='monitor',
                       help='Disable progress updates')

    args = parser.parse_args()

    # Handle special commands
    if args.status:
        show_status()
        return

    if args.clear_cache:
        from core.ml.parallel_trainer import ParallelTrainer
        trainer = ParallelTrainer()
        trainer.clear_feature_cache()
        return

    # Determine pairs to train
    if args.symbols:
        pairs = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.tier:
        pairs = SYMBOL_TIERS[args.tier]
    elif args.all:
        pairs = None  # ParallelTrainer will get all pairs with data
    else:
        # No target specified, show help
        parser.print_help()
        print("\nUse --status to see current training status")
        print("Use --all to train all pairs with data")
        return

    # Initialize GPU config
    print("Configuring GPU for maximum performance...")
    from core.ml.gpu_config import optimize_for_training
    optimize_for_training()

    # Create trainer
    from core.ml.parallel_trainer import ParallelTrainer

    trainer = ParallelTrainer(
        cpu_workers=args.cpu_workers,
        gpu_workers=args.gpu_workers,
        max_samples=args.max_samples,
    )

    # Run training
    results = trainer.train_all(
        pairs=pairs,
        skip_trained=not args.no_skip,
        monitor=args.monitor,
        dry_run=args.dry_run,
    )

    # Final summary
    if results:
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully trained: {len(results)} pairs")

        # Show accuracy summary
        accuracies = []
        for pair, res in results.items():
            for target, metrics in res.items():
                if target == 'target_direction_1':
                    accuracies.append(metrics['accuracy'])

        if accuracies:
            print(f"Direction_1 accuracy: {min(accuracies):.2%} - {max(accuracies):.2%}")
            print(f"Average: {sum(accuracies)/len(accuracies):.2%}")


if __name__ == "__main__":
    main()
