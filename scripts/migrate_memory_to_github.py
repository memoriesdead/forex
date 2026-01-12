#!/usr/bin/env python3
"""
Migrate Local MCP Memory to GitHub
Compacts and uploads all local memory-keeper context to GitHub repository
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_MEMORY_REPO")  # format: username/repo-name
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

# Local export file
EXPORT_FILE = Path(r"C:\Users\kevin\AppData\Local\Temp\memory-keeper-export-97e9dae9.json")


def compact_context_items(items):
    """Compact context items by removing unnecessary fields and grouping by category"""
    compacted = {
        "decisions": [],
        "progress": [],
        "notes": [],
        "errors": [],
        "warnings": []
    }

    for item in items:
        compact_item = {
            "key": item["key"],
            "value": item["value"],
            "priority": item["priority"],
            "created": item["created_at"]
        }

        category = item.get("category", "notes")
        if category not in compacted:
            compacted[category] = []

        compacted[category].append(compact_item)

    return compacted


def create_github_file(path, content, message):
    """Create or update a file in GitHub repository"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("[ERROR] GitHub credentials not configured in .env")
        print("Add: GITHUB_TOKEN and GITHUB_MEMORY_REPO")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Check if file exists
    response = requests.get(url, headers=headers)
    sha = None
    if response.status_code == 200:
        sha = response.json().get("sha")

    # Prepare content
    import base64
    content_bytes = content.encode('utf-8')
    content_b64 = base64.b64encode(content_bytes).decode('utf-8')

    # Create/update file
    data = {
        "message": message,
        "content": content_b64,
        "branch": GITHUB_BRANCH
    }

    if sha:
        data["sha"] = sha

    response = requests.put(url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        print(f"[SUCCESS] Uploaded: {path}")
        return True
    else:
        print(f"[ERROR] Failed to upload {path}: {response.status_code}")
        print(response.json())
        return False


def migrate_to_github():
    """Main migration function"""
    print("=" * 60)
    print("Migrating Local Memory to GitHub")
    print("=" * 60)
    print()

    # Load local export
    if not EXPORT_FILE.exists():
        print(f"[ERROR] Export file not found: {EXPORT_FILE}")
        return False

    print(f"[1/5] Loading local memory export...")
    with open(EXPORT_FILE, 'r', encoding='utf-8') as f:
        export_data = json.load(f)

    session = export_data["session"]
    items = export_data["contextItems"]
    checkpoints = export_data["checkpoints"]

    print(f"  Found: {len(items)} context items, {len(checkpoints)} checkpoints")

    # Compact data
    print(f"[2/5] Compacting context data...")
    compacted = compact_context_items(items)

    # Create session summary
    session_summary = {
        "session_id": session["id"],
        "name": session["name"],
        "description": session["description"],
        "date": datetime.fromisoformat(session["created_at"].replace("Z", "+00:00")).strftime("%Y-%m-%d"),
        "items_count": len(items),
        "checkpoints": [
            {
                "name": cp["name"],
                "description": cp["description"],
                "created": cp["created_at"]
            }
            for cp in checkpoints
        ],
        "context": compacted
    }

    # Calculate stats
    stats = {
        "decisions": len(compacted["decisions"]),
        "progress": len(compacted["progress"]),
        "notes": len(compacted["notes"]),
        "errors": len(compacted["errors"]),
        "warnings": len(compacted["warnings"])
    }

    print(f"  Compacted stats: {stats}")

    # Upload to GitHub
    print(f"[3/5] Uploading to GitHub repository: {GITHUB_REPO}...")

    # Create session file
    session_date = session_summary["date"]
    session_file = f"sessions/{session_date}-oracle-cloud-storage-integration.json"
    session_content = json.dumps(session_summary, indent=2, ensure_ascii=False)

    if not create_github_file(
        session_file,
        session_content,
        f"Add session: {session['name']} ({session_date})"
    ):
        return False

    # Create/update README
    print(f"[4/5] Creating README...")
    readme_content = f"""# Forex Trading ML - Memory Storage

## Project
High-frequency forex trading system with ML price prediction

## Sessions

### {session_date} - {session['name']}
- **Items**: {len(items)}
- **Decisions**: {stats['decisions']}
- **Progress**: {stats['progress']}
- **Notes**: {stats['notes']}

## Storage Structure

```
sessions/
└── {session_date}-oracle-cloud-storage-integration.json
```

## Latest Context

**Oracle Cloud Integration Complete**
- Instance: 89.168.65.47
- Storage: 42GB available
- Isolation: /home/ubuntu/projects/forex/
- MCP servers migrated to Oracle Cloud
- Memory storage migrated to GitHub (this repo)

## Key Decisions

{chr(10).join(f"- {item['key']}: {item['value'][:100]}..." for item in compacted['decisions'][:5])}

---
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Managed by: forex trading ML project*
"""

    if not create_github_file("README.md", readme_content, "Update README with session data"):
        return False

    # Create index
    print(f"[5/5] Creating index...")
    index = {
        "project": "forex-trading-ml",
        "updated": datetime.now().isoformat(),
        "sessions": [
            {
                "date": session_date,
                "name": session["name"],
                "file": session_file,
                "items": len(items),
                "stats": stats
            }
        ]
    }

    index_content = json.dumps(index, indent=2, ensure_ascii=False)
    if not create_github_file("index.json", index_content, "Update index"):
        return False

    print()
    print("=" * 60)
    print("[SUCCESS] Migration complete!")
    print("=" * 60)
    print()
    print(f"Repository: https://github.com/{GITHUB_REPO}")
    print(f"Session file: {session_file}")
    print(f"Total items migrated: {len(items)}")
    print()

    return True


def cleanup_local():
    """Clean up local MCP files after successful migration"""
    print("=" * 60)
    print("Cleaning up local MCP files")
    print("=" * 60)
    print()

    response = input("Delete local MCP memory files? This is irreversible. (yes/no): ")
    if response.lower() != "yes":
        print("[INFO] Cleanup cancelled")
        return

    # Paths to clean
    cleanup_paths = [
        Path(r"C:\Users\kevin\.claude\memory"),
        Path(r"C:\Users\kevin\forex\.claude-mem"),
        EXPORT_FILE
    ]

    for path in cleanup_paths:
        if path.exists():
            print(f"[INFO] Cleaning: {path}")
            if path.is_file():
                path.unlink()
                print(f"  Deleted file: {path}")
            elif path.is_dir():
                import shutil
                # Only delete if it's a memory directory
                if "memory" in str(path) or "claude-mem" in str(path):
                    shutil.rmtree(path)
                    print(f"  Deleted directory: {path}")
        else:
            print(f"  Not found: {path}")

    print()
    print("[SUCCESS] Local cleanup complete")
    print()


def main():
    if not GITHUB_TOKEN:
        print("[ERROR] GITHUB_TOKEN not set in .env")
        print("\nTo setup:")
        print("1. Create GitHub repo (e.g., 'forex-memory')")
        print("2. Generate PAT: https://github.com/settings/tokens")
        print("3. Add to .env:")
        print("   GITHUB_TOKEN=ghp_your_token_here")
        print("   GITHUB_MEMORY_REPO=username/forex-memory")
        return 1

    if not GITHUB_REPO:
        print("[ERROR] GITHUB_MEMORY_REPO not set in .env")
        print("Add: GITHUB_MEMORY_REPO=username/forex-memory")
        return 1

    # Migrate
    if not migrate_to_github():
        print("[ERROR] Migration failed")
        return 1

    # Ask to cleanup
    cleanup_local()

    return 0


if __name__ == "__main__":
    sys.exit(main())
