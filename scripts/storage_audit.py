#!/usr/bin/env python3
"""
Audit Oracle Cloud storage capacity for 85-pair download system.
"""

# Oracle Cloud Free Tier
TOTAL_STORAGE_GB = 45
USED_STORAGE_GB = 5.9
AVAILABLE_GB = 39

# Current data (10 pairs, 3 dates)
CURRENT_FILES = 30
CURRENT_SIZE_MB = 253

# Calculated metrics
CURRENT_PER_FILE_MB = CURRENT_SIZE_MB / CURRENT_FILES  # 8.4MB avg
CURRENT_PER_DATE_MB = CURRENT_SIZE_MB / 3  # 84MB per date (10 pairs)

# Projected for 85 pairs
PAIRS_CURRENT = 10
PAIRS_TARGET = 85
SCALE_FACTOR = PAIRS_TARGET / PAIRS_CURRENT  # 8.5x

# Estimates based on actual data
# Monday 1/6/2026 (full trading day): 10 pairs = ~66.5MB raw
FULL_DAY_10_PAIRS_MB = 66.5
FULL_DAY_85_PAIRS_RAW_MB = FULL_DAY_10_PAIRS_MB * SCALE_FACTOR
FULL_DAY_85_PAIRS_CLEANED_MB = FULL_DAY_85_PAIRS_RAW_MB  # Similar size
FULL_DAY_TOTAL_MB = FULL_DAY_85_PAIRS_RAW_MB + FULL_DAY_85_PAIRS_CLEANED_MB

# Convert to GB
FULL_DAY_TOTAL_GB = FULL_DAY_TOTAL_MB / 1024

# Weekend estimate (much smaller - 1/5/2026 example)
WEEKEND_10_PAIRS_MB = 25  # Estimated based on EURUSD 259KB
WEEKEND_85_PAIRS_MB = WEEKEND_10_PAIRS_MB * SCALE_FACTOR
WEEKEND_TOTAL_GB = (WEEKEND_85_PAIRS_MB * 2) / 1024  # raw + cleaned

# Weekly average (5 weekdays + 2 weekend days)
WEEKLY_MB = (FULL_DAY_TOTAL_MB * 5) + (WEEKEND_TOTAL_GB * 1024 * 2)
WEEKLY_GB = WEEKLY_MB / 1024
DAILY_AVG_GB = WEEKLY_GB / 7

print("=" * 70)
print("ORACLE CLOUD STORAGE AUDIT - 85 FOREX PAIRS")
print("=" * 70)

print("\nCURRENT USAGE:")
print(f"  Total storage: {TOTAL_STORAGE_GB}GB")
print(f"  Used (system): {USED_STORAGE_GB}GB")
print(f"  Available:     {AVAILABLE_GB}GB")
print(f"  Current forex: {CURRENT_SIZE_MB}MB ({CURRENT_FILES} files, 10 pairs, 3 dates)")

print("\nESTIMATED DATA VOLUME (85 PAIRS):")
print(f"  Full weekday:  {FULL_DAY_TOTAL_MB:.0f}MB = {FULL_DAY_TOTAL_GB:.2f}GB")
print(f"    - Raw data:  {FULL_DAY_85_PAIRS_RAW_MB:.0f}MB ({PAIRS_TARGET} pairs)")
print(f"    - Cleaned:   {FULL_DAY_85_PAIRS_CLEANED_MB:.0f}MB ({PAIRS_TARGET} pairs)")
print(f"  Weekend day:   {WEEKEND_85_PAIRS_MB * 2:.0f}MB = {WEEKEND_TOTAL_GB:.2f}GB")
print(f"  Weekly total:  {WEEKLY_GB:.2f}GB")
print(f"  Daily average: {DAILY_AVG_GB:.2f}GB")

print("\nCRON SCHEDULE ANALYSIS:")
print("  Runs every 6 hours: 0, 6, 12, 18 UTC")
print("  Downloads previous day if available")
print("  Data stays on Oracle until manual transfer")

print("\nWORST-CASE SCENARIOS:")

# Scenario 1: User doesn't transfer for a week
days_no_transfer = 7
worst_case_1_gb = days_no_transfer * DAILY_AVG_GB
print(f"\n  1. No transfer for {days_no_transfer} days:")
print(f"     Data accumulated: {worst_case_1_gb:.2f}GB")
print(f"     Storage used: {USED_STORAGE_GB + worst_case_1_gb:.2f}GB / {TOTAL_STORAGE_GB}GB")
print(f"     Remaining: {AVAILABLE_GB - worst_case_1_gb:.2f}GB")
if worst_case_1_gb < AVAILABLE_GB:
    print(f"     Status: SAFE - {((AVAILABLE_GB - worst_case_1_gb) / DAILY_AVG_GB):.0f} more days capacity")
else:
    print(f"     Status: EXCEEDS FREE TIER")

# Scenario 2: User doesn't transfer for a month
days_no_transfer = 30
worst_case_2_gb = days_no_transfer * DAILY_AVG_GB
print(f"\n  2. No transfer for {days_no_transfer} days:")
print(f"     Data accumulated: {worst_case_2_gb:.2f}GB")
print(f"     Storage used: {USED_STORAGE_GB + worst_case_2_gb:.2f}GB / {TOTAL_STORAGE_GB}GB")
print(f"     Remaining: {AVAILABLE_GB - worst_case_2_gb:.2f}GB")
if worst_case_2_gb < AVAILABLE_GB:
    print(f"     Status: SAFE")
else:
    print(f"     Status: EXCEEDS FREE TIER by {worst_case_2_gb - AVAILABLE_GB:.2f}GB")

# Calculate max safe storage time
max_days = AVAILABLE_GB / DAILY_AVG_GB
print(f"\n  3. Maximum safe storage:")
print(f"     Can store {max_days:.0f} days without transfer")
print(f"     Before reaching {TOTAL_STORAGE_GB}GB limit")

print("\nRECOMMENDATIONS:")

if max_days >= 30:
    print("  [OK] Free tier can handle 85 pairs")
    print(f"  [OK] Can store {max_days:.0f} days without transfer")
    print("  [OK] Transfer weekly recommended")
elif max_days >= 7:
    print("  [WARN] Free tier can handle 85 pairs with regular transfers")
    print(f"  [WARN] MUST transfer at least weekly ({max_days:.0f} days max)")
    print("  [WARN] Recommend transfer every 3-4 days")
else:
    print("  [FAIL] Free tier TOO SMALL for 85 pairs")
    print(f"  [FAIL] Can only store {max_days:.0f} days")
    print(f"  [FAIL] MUST transfer daily or reduce pairs")

print("\nAUTO-DELETE STRATEGY:")
if max_days < 7:
    print("  CRITICAL: Enable auto-delete after download")
    print("  Modify cron to download + transfer + delete in one run")
elif max_days < 14:
    print("  IMPORTANT: Transfer at least twice weekly")
    print("  Or enable auto-delete after download")
else:
    print("  SAFE: Weekly transfers sufficient")
    print("  Current manual transfer approach works")

# File count analysis
files_per_day = PAIRS_TARGET * 2  # raw + cleaned
max_files = int(max_days * files_per_day)
print(f"\nFILE COUNT:")
print(f"  Per day: {files_per_day} files ({PAIRS_TARGET} pairs × 2)")
print(f"  Max storable: {max_files} files ({max_days:.0f} days)")

print("\n" + "=" * 70)
print(f"VERDICT: {'SAFE' if max_days >= 7 else 'UNSAFE'} for free tier with {'weekly' if max_days >= 7 else 'daily'} transfers")
print("=" * 70)
