---
description: Update CLAUDE.md when significant code changes occur
allowedTools: [Bash, Read, Write, Edit, Grep, Glob]
---

# Update CLAUDE.md

Analyze recent code changes and update CLAUDE.md to reflect significant architectural or workflow changes.

## 1. Identify Changed Files
- Use `git diff` to find changed files (or specify commit range)
- Identify the scope: new features, refactoring, configuration changes, etc.
- Commands: `git status`, `git diff --name-only`, `git log --oneline -10`

## 2. Analyze Change Significance
Determine if changes warrant CLAUDE.md updates by checking for:

### High-Priority Changes (always update):
- **New build commands**: package.json scripts, Makefiles, build configs
- **New test commands**: test frameworks, test runners, test scripts
- **Architecture changes**: new directories, module reorganization, design patterns
- **New dependencies**: major libraries or frameworks added
- **Development workflow**: new tools, linters, formatters, pre-commit hooks
- **Configuration files**: new .env requirements, config structure changes
- **API changes**: new endpoints, breaking changes, authentication methods

### Medium-Priority Changes (update if substantial):
- **New features**: significant new functionality
- **Directory structure**: major reorganization
- **Development setup**: Docker, database, environment setup changes
- **Documentation standards**: new doc requirements or formats

### Low-Priority Changes (usually skip):
- Bug fixes without workflow impact
- Minor refactoring
- Code style changes
- Single file updates

## 3. Read Current CLAUDE.md
- Read the existing CLAUDE.md file
- Identify which sections exist
- Understand current structure and content

## 4. Determine What to Update
Based on changes, identify which CLAUDE.md sections need updates:

- **Commands Section**: New build, test, lint, or run commands
- **Architecture Section**: High-level structure changes
- **Development Setup**: Environment or dependency changes
- **Code Patterns**: New conventions or patterns introduced
- **Configuration**: New settings or environment variables

## 5. Generate Updates
For each section that needs updating:

- **Be concise**: Focus on "what" and "why", not implementation details
- **Be specific**: Include exact commands with examples
- **Be architectural**: Explain high-level structure, not file-by-file details
- **Avoid repetition**: Don't duplicate information already in README
- **Future-focused**: Help future Claude instances be productive quickly

## 6. Present Changes
Show the user:
- Summary of what changed in the codebase
- Proposed CLAUDE.md updates with clear before/after
- Rationale for each update

Ask for confirmation before applying changes.

## Output Format

### Changes Detected
- List significant changes found
- Categorize by type (commands, architecture, config, etc.)

### Proposed CLAUDE.md Updates
For each section:
```markdown
## Section Name

[Show proposed additions or changes]
```

### Rationale
Explain why these updates help future Claude Code instances
