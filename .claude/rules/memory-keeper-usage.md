# Memory Keeper MCP Server Usage

## CRITICAL: Project Isolation

**This is the FOREX project. Memory-keeper is shared across 6 projects.**

| Setting | Value | NEVER Use |
|---------|-------|-----------|
| projectDir | `C:\Users\kevin\forex` | kalshi, bitcoin, polymarket, alpaca, paymentcrypto |
| project | `forex` | Any other project name |
| channel prefix | `forex-*` | Other project prefixes |

**Before ANY memory-keeper call:**
1. Verify projectDir is `C:\Users\kevin\forex`
2. Verify you're not accessing another project's context
3. Use forex-specific channels when organizing data

**Violations contaminate shared infrastructure and break all 6 projects.**

---

## Mandatory Session Management

**ALWAYS start a session:**
- Tool: `mcp__memory-keeper__context_session_start`
- Set `projectDir: C:\Users\kevin\forex`
- Use descriptive name and description
- Do this at the beginning of every conversation

**During work - Save proactively:**
- Don't batch saves at end of session
- Save immediately after important decisions or completions
- Use appropriate categories and priorities

## Categories and When to Use

- `task`: Action items, TODOs, work in progress
- `decision`: Architecture choices, design patterns, technology selections
- `progress`: Completed milestones, achievements, successful implementations
- `error`: Problems encountered and their solutions
- `note`: General information, configurations, references
- `warning`: Gotchas, cautions, things to watch out for

## Priority Levels

- `high`: Critical information that impacts multiple areas (API keys, core architecture, breaking changes)
- `normal`: Important but not critical (feature implementations, refactorings)
- `low`: Nice-to-have, reference information (code comments, documentation links)

## Required Saves

**ALWAYS save these to memory:**
1. API endpoint configurations
2. Data source decisions and changes
3. Training hyperparameters and results
4. Error patterns and their solutions
5. Environment variable additions/changes
6. Architecture or design changes
7. Performance optimization decisions

## Knowledge Graph Links

**Link related items using:**
- `depends_on`: When code/model depends on data/config
- `implements`: When code implements a design decision
- `references`: When documentation references code
- `blocks`: When an error blocks progress
- `related_to`: General relationships

**Example linking pattern:**
```
data_source → (depends_on) → processing_script → (depends_on) → cleaned_data → (depends_on) → model_training
```

## Checkpoints

**Create checkpoints before:**
- Major refactorings
- Experimental changes
- Model training runs
- Data pipeline modifications

Use: `mcp__memory-keeper__context_checkpoint`

## Search and Retrieval

**Use semantic search for questions:**
- "What data sources are we using?"
- "How did we solve the API timeout issue?"
- Tool: `mcp__memory-keeper__context_semantic_search`

**Use filtered search for specific queries:**
- Filter by category, channel, priority, date range
- Tool: `mcp__memory-keeper__context_search`

**Search across all sessions:**
- Project-wide context retrieval
- Tool: `mcp__memory-keeper__context_search_all`

## Channels

- Auto-derived from git branches
- Organizes context by feature/topic
- Use default channel unless working on specific feature branch
