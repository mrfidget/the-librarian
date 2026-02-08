#!/usr/bin/env python3
"""
The Librarian - Main CLI Application
"""
import argparse
import sys
from pathlib import Path

from src.orchestrator import Orchestrator
from src.config import get_config


def process_command(args):
    """Handle process command."""
    orchestrator = Orchestrator()

    # Read URLs from file or command line
    urls = []

    if args.url_file:
        with open(args.url_file, 'r') as f:
            # Skip comments and blank lines
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    elif args.urls:
        urls = args.urls
    else:
        print("Error: Either --url-file or URLs must be provided")
        return 1

    print(f"Processing {len(urls)} URL(s)...")
    processed = orchestrator.process_urls(urls, clean_after=not args.keep_staging)

    print(f"\nSuccessfully processed {processed} files")
    return 0


def score_to_stars(score: float) -> str:
    """
    Convert a numeric similarity score to a star rating.
    
    Args:
        score: Similarity score between 0.0 and 1.0
        
    Returns:
        Star rating string (e.g., "★★★★★" or "★★☆☆☆")
    """
    if score >= 0.8:
        return "★★★★★"
    elif score >= 0.6:
        return "★★★★☆"
    elif score >= 0.4:
        return "★★★☆☆"
    elif score >= 0.2:
        return "★★☆☆☆"
    else:
        return "★☆☆☆☆"


def format_results_table(results, threshold=0.25):
    """
    Format search results as a compact table with Ms. Clarke's commentary.
    
    Args:
        results: List of SearchResult objects
        threshold: Minimum score to display (default 0.25)
        
    Returns:
        Formatted string with flavor text and table
    """
    from pathlib import Path
    
    # Filter by threshold
    filtered = [r for r in results if r.score >= threshold]
    
    if not filtered:
        return (
            "\n*Ms. Clarke peers over her glasses disapprovingly*\n\n"
            '"I\'ve searched the entire collection. Nothing even remotely\n'
            'relevant. Are you certain you phrased that correctly?"\n'
        )
    
    # Determine Ms. Clarke's mood based on top result quality
    top_score = filtered[0].score
    
    if top_score >= 0.8:
        flavor = (
            "\n*Ms. Clarke returns promptly, looking rather pleased*\n\n"
            '"Ah yes. I thought so. Here\'s what you\'re after:"\n'
        )
    elif top_score >= 0.5:
        flavor = (
            "\n*Ms. Clarke adjusts her reading glasses*\n\n"
            '"I believe these should suffice:"\n'
        )
    else:
        flavor = (
            "\n*Ms. Clarke sighs and sets down a small stack*\n\n"
            '"Well. These are... adjacent to your request, I suppose.\n'
            'Perhaps you could be more specific next time?"\n'
        )
    
    # Build table
    lines = [flavor]
    lines.append(f"\n{'Rank':<6} {'Stars':<10} {'Type':<8} {'Filename':<30} Description")
    lines.append("-" * 100)
    
    for i, r in enumerate(filtered, 1):
        stars = score_to_stars(r.score)
        # Use original filename from file_path (staging path) instead of library checksum
        filename = Path(r.file_path).name
        # Truncate long filenames and descriptions
        filename = filename[:28] + '..' if len(filename) > 30 else filename
        desc = r.description[:48] + '...' if len(r.description) > 50 else r.description
        
        lines.append(f"{i:<6} {stars:<10} {r.file_type.value:<8} {filename:<30} {desc}")
    
    lines.append("")
    lines.append(f"*{len(filtered)} item(s) found. Files are in /data/library/*")
    lines.append("")
    
    return "\n".join(lines)


def query_command(args):
    """Handle query command with Ms. Clarke's personality."""
    config = get_config()
    orchestrator = Orchestrator()

    if args.interactive:
        # Welcome message
        print("\n" + "="*80)
        print("The Librarian — Ms. Clarke, Head Librarian")
        print("="*80)
        print("\n*Ms. Clarke adjusts her reading glasses and looks up from her desk*\n")
        print('"Good afternoon. State your inquiry clearly, please.')
        print('And do remember: I expect precision. None of this')
        print('\'find me stuff about things\' nonsense."\n')
        print('Type \'quit\' or \'exit\' to leave.\n')
        print("="*80 + "\n")

        while True:
            try:
                query = input("Ask Ms. Clarke: ").strip()

                if query.lower() in ('quit', 'exit', 'q'):
                    print("\n*Ms. Clarke nods curtly*\n")
                    print('"Very well. Do try to return your materials on time."')
                    print()
                    break

                if not query:
                    continue

                results = orchestrator.search(query, limit=args.limit)

                # Format and display with threshold filtering
                output = format_results_table(results, threshold=config.search_threshold)
                print(output)

            except KeyboardInterrupt:
                print("\n\n*Ms. Clarke raises an eyebrow*\n")
                print('"Leaving so soon? Very well."')
                print()
                break
            except EOFError:
                break
    else:
        # Single query mode (no personality, just results)
        if not args.query:
            print("Error: --query required for non-interactive mode")
            return 1

        results = orchestrator.search(args.query, limit=args.limit)
        output = format_results_table(results, threshold=config.search_threshold)
        print(output)

    return 0


def backup_command(args):
    """Handle backup command."""
    orchestrator = Orchestrator()

    if args.restore:
        backup_path = Path(args.restore)
        if not backup_path.exists():
            print(f"Error: Backup path {backup_path} does not exist")
            return 1

        print(f"Restoring from {backup_path}...")
        success = orchestrator.restore_data(backup_path)

        if success:
            print("Restore completed successfully")
            return 0
        else:
            print("Restore failed")
            return 1
    else:
        print("Creating backup...")
        success = orchestrator.backup_data()

        if success:
            print("Backup completed successfully")
            return 0
        else:
            print("Backup failed")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="The Librarian - AI-powered document management system"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process URLs')
    process_parser.add_argument('urls', nargs='*', help='URLs to process')
    process_parser.add_argument('--url-file', '-f', help='File containing URLs (one per line)')
    process_parser.add_argument('--keep-staging', action='store_true',
                                help='Keep staging files after processing')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query documents')
    query_parser.add_argument('query', nargs='?', help='Search query')
    query_parser.add_argument('--interactive', '-i', action='store_true',
                              help='Interactive query mode')
    query_parser.add_argument('--limit', '-l', type=int, default=10,
                              help='Maximum number of results (default: 10)')

    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup/restore data')
    backup_parser.add_argument('--restore', '-r', help='Restore from backup path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'process':
        return process_command(args)
    elif args.command == 'query':
        return query_command(args)
    elif args.command == 'backup':
        return backup_command(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())