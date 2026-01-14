# Code Style and Quality

## Based on .claude/settings.json

**Vibe Coding Philosophy:**
- Move fast, iterate quickly
- Production-ready on first implementation
- No placeholders or TODOs
- Complete implementations only
- Modern patterns and best practices

## Code Quality Standards

**DO:**
- Write production-ready code immediately
- Use modern Python patterns
- Type hints for function signatures
- Clear, descriptive variable names
- Error handling at system boundaries (user input, external APIs)

**DON'T:**
- Add TODO comments or placeholders
- Over-engineer solutions
- Add features beyond requirements
- Create unnecessary abstractions
- Add error handling for impossible scenarios

## Simplicity First

**Avoid:**
- Helper functions for one-time operations
- Premature abstractions
- Designing for hypothetical future requirements
- Feature flags or backwards-compatibility hacks
- Renaming unused variables with `_`

**Prefer:**
- Three similar lines over premature abstraction
- Deleting unused code completely
- Trust internal code and framework guarantees
- Minimum complexity needed for current task

## When to Add Comments

- Only where logic isn't self-evident
- Complex algorithms or math
- Business logic that seems arbitrary
- NOT for obvious operations
- NOT for code you didn't change

## Security

**Always check for:**
- Command injection vulnerabilities
- XSS in any web interfaces
- SQL injection if using databases
- Unsafe deserialization
- Exposed credentials

**If you write insecure code:**
- Fix it immediately
- Document the vulnerability in memory (category: error, priority: high)
- Save the solution for future reference

## File Organization

**Root directory:**
- Keep clean (only folders and .env files)
- No loose scripts or temporary files
- Organize code in appropriate subdirectories

**Temporary files:**
- Delete after use
- Don't commit to git
