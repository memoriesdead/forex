# Data Handling Rules

## Data Directory Structure

- `data/`: Raw data from sources (DO NOT MODIFY)
- `data_cleaned/`: Processed, cleaned data ready for ML
- Keep raw data immutable for reproducibility

## Data Sources

**TrueFX:**
- Real-time tick data
- High frequency, low latency
- Save API endpoints to memory (category: decision, priority: high)

**Dukascopy:**
- Historical forex data
- Use for backtesting and training
- Document download scripts and schedules in memory

**Hostinger:**
- Custom data storage/hosting
- Save connection details to memory (category: note, priority: high)

## Data Processing Pipeline

**When processing data:**
1. Save processing decision to memory (category: decision)
2. Document transformation steps
3. Link: `raw_data` → `processing_script` → `cleaned_data`
4. Validate data quality and save results (category: progress)

**Always document:**
- Cleaning steps applied
- Validation rules used
- Data quality issues found
- Transformations performed

## Environment Variables

**Structure to track in memory:**
- API keys and endpoints
- Database credentials
- Service URLs
- Feature flags

**Files:**
- `.env`: Main configuration
- `.env.backup`: Backup copy
- `.env_hostinger`: Hostinger-specific

**NEVER:**
- Commit .env files to git
- Share credentials in memory (save structure only)
- Hard-code credentials in scripts
