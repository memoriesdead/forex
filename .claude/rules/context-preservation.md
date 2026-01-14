# Context Preservation for Vibe Coding

## The Problem

**Vibe coding = Move fast + Long sessions = High token usage = Context compaction**

When compaction happens:
- Conversation history is summarized/lost
- Only memory-keeper MCP server persists
- Context not saved to memory = gone forever

## The Solution: Aggressive Proactive Saving

### Save Immediately Pattern

**After every significant action, IMMEDIATELY save:**

```
Decision made → mcp__memory-keeper__context_save (category: decision)
Task completed → mcp__memory-keeper__context_save (category: progress)
Error solved → mcp__memory-keeper__context_save (category: error)
Config changed → mcp__memory-keeper__context_save (category: note, priority: high)
```

**DON'T batch saves - save one at a time as things happen**

### Critical Information That MUST Be Saved

**Priority: high (survives compaction first):**

1. **Current State**
   - What you're working on RIGHT NOW
   - Next steps planned
   - Blockers or issues

2. **Recent Decisions**
   - Why something was done a certain way
   - Alternatives considered and rejected
   - Trade-offs made

3. **File Locations**
   - Where important scripts are
   - Where data is stored
   - Where models are saved

4. **Configuration**
   - API endpoints in use
   - Environment variable structure
   - Connection details (not credentials!)

5. **Failed Attempts**
   - What was tried and didn't work
   - Why it failed
   - What to avoid

6. **Working Solutions**
   - What fixed a problem
   - How errors were resolved
   - Performance optimizations that worked

### Compaction Warning Signs

**Watch for these - then act immediately:**

- Token usage > 100,000
- Conversation > 50 messages in one task
- Multiple complex tasks in one session
- Long code generation or debugging sessions

**When you see these, run:**
```
mcp__memory-keeper__context_prepare_compaction
```

### Checkpoint Strategy

**Checkpoint at these points:**

1. **Task boundaries**
   - Completed a feature → checkpoint
   - About to start new feature → checkpoint
   - Switching contexts → checkpoint

2. **Risk points**
   - Before refactoring → checkpoint
   - Before experimental changes → checkpoint
   - Before model training → checkpoint

3. **Time intervals**
   - Every 50-100 messages → checkpoint
   - Every hour of work → checkpoint
   - Before taking a break → checkpoint

**Checkpoint command:**
```
mcp__memory-keeper__context_checkpoint
- name: descriptive_name_with_timestamp
- includeGitStatus: true
- includeFiles: true
```

### Knowledge Graph Linking Strategy

**Link aggressively to preserve relationships:**

```
Every script → depends_on → data it uses
Every model → depends_on → training script
Every solution → related_to → error it solved
Every decision → implements → architecture choice
```

**Why:** If conversation is compacted, links preserve how things relate

### Recovery Protocol

**If context seems lost after compaction:**

**Step 1: Search recent context**
```
mcp__memory-keeper__context_search_all
- query: "current work in progress"
- createdAfter: "2 hours ago"
- sort: "created_desc"
- limit: 20
```

**Step 2: Check timeline**
```
mcp__memory-keeper__context_timeline
- groupBy: "hour"
- relativeTime: "today"
- includeItems: true
```

**Step 3: Restore checkpoint if needed**
```
mcp__memory-keeper__context_restore_checkpoint
- name: "most_recent_checkpoint_name"
```

**Step 4: View diff since last checkpoint**
```
mcp__memory-keeper__context_diff
- since: "checkpoint_name"
- includeValues: true
```

### Batch Operations for Efficiency

**When saving multiple related items:**

```
mcp__memory-keeper__context_batch_save
items: [
  {key: "config_api_endpoint", value: "...", category: "note", priority: "high"},
  {key: "config_timeout", value: "...", category: "note", priority: "normal"},
  {key: "config_retry_logic", value: "...", category: "note", priority: "normal"}
]
```

**Better than 3 separate saves**

### Session Continuity

**At session end, ALWAYS save:**

1. Current work status (what's done, what's pending)
2. Next steps to take
3. Any blockers or open questions
4. Recent file changes made

**At next session start, ALWAYS search:**

```
mcp__memory-keeper__context_search
- query: "next steps OR current work OR pending"
- sort: "created_desc"
- limit: 10
```

### Priority Levels Strategy

**Use priority to control what survives compaction:**

- `priority: high` = Critical, must survive compaction
- `priority: normal` = Important, should survive
- `priority: low` = Nice-to-have, can be compressed

**High priority examples:**
- Active work-in-progress
- Unresolved errors
- Recent architecture decisions
- API configurations
- Model training results

**Normal priority examples:**
- Completed tasks
- Resolved errors
- Code patterns used
- Optimization notes

**Low priority examples:**
- Random observations
- Temporary notes
- Experimental ideas that didn't pan out

### Channels for Organization

**Use channels to organize context by topic:**

- Auto-derived from git branches (default)
- Manual override for specific features
- Search within channel to find related context

**Example:**
```
mcp__memory-keeper__context_save
- key: "data_pipeline_optimization"
- value: "..."
- channel: "performance_improvements"
- category: "decision"
- priority: "high"
```

### Watchers for Critical Changes

**Set up watchers for important patterns:**

```
mcp__memory-keeper__context_watch
- action: "create"
- filters: {
    categories: ["error", "decision"],
    priorities: ["high"]
  }
```

**Poll periodically to catch critical updates**

## Rules of Thumb

1. **Save early, save often** - Don't wait
2. **High priority for active work** - Current state must survive
3. **Link everything** - Relationships preserve understanding
4. **Checkpoint at boundaries** - Task switches, risk points
5. **Prepare before compaction** - Watch token usage
6. **Search at session start** - Recover context immediately
7. **Batch related saves** - More efficient
8. **Use descriptive keys** - Easy to search later

## Anti-Patterns to Avoid

**DON'T:**
- Wait until end of session to save (too late)
- Save without priority tags (everything becomes equal)
- Save without categories (can't filter later)
- Skip linking (relationships get lost)
- Ignore token usage (compaction surprises you)
- Save with vague keys (can't search effectively)
- Forget to checkpoint (no recovery point)

**DO:**
- Save immediately as things happen
- Tag with appropriate priority and category
- Link related items in knowledge graph
- Monitor token usage proactively
- Use descriptive, searchable keys
- Checkpoint regularly at boundaries
