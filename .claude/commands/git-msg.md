---
description: Generate a short but descriptive git commit message
allowedTools: [Bash, Read, Grep]
---

# Git Message Generator

Analyze current git changes and generate a concise, descriptive commit message.

## Step 1: Analyze Changes

### Check Repository State

```bash
# Get current branch
git branch --show-current

# Check for staged changes
git diff --cached --name-status

# Check for unstaged changes
git diff --name-status

# Get file statistics
git diff --cached --stat
git diff --stat
```

### Determine Change Scope

Identify what files are affected:
- **New files**: Added functionality or resources
- **Modified files**: Updates, fixes, or improvements
- **Deleted files**: Cleanup or removal
- **File types**: Code, tests, docs, config

## Step 2: Analyze Change Content

### Get Detailed Diffs

```bash
# View staged changes
git diff --cached

# If no staged changes, view unstaged
git diff
```

### Categorize Changes

Determine the primary change type:

#### 1. **Feature** (new functionality)
- New functions, classes, or modules added
- New API endpoints
- New user-facing features
- Pattern: `^\+.*function|class|def|export.*function`

#### 2. **Fix** (bug fixes)
- Bug corrections
- Error handling improvements
- Edge case fixes
- Pattern: `fix|bug|error|null check|validation`

#### 3. **Refactor** (code improvement, no behavior change)
- Code reorganization
- Performance optimization
- Simplification
- Pattern: Large deletions + additions in same files

#### 4. **Docs** (documentation only)
- README updates
- Comment additions
- API documentation
- Pattern: Only `.md` files or docstrings changed

#### 5. **Test** (test code)
- New tests
- Test updates
- Pattern: `test_|.test.|.spec.|__tests__`

#### 6. **Chore** (maintenance tasks)
- Dependency updates
- Configuration changes
- Build script updates
- Pattern: `package.json|requirements.txt|*.config.*`

#### 7. **Style** (formatting, no logic change)
- Code formatting
- Linting fixes
- Whitespace changes
- Pattern: Only whitespace/formatting diffs

## Step 3: Generate Commit Message

### Message Format

Use this concise format:

```
<type>: <short description>
```

**Type prefix options:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation only
- `test:` - Test code
- `chore:` - Maintenance tasks
- `style:` - Formatting changes
- `perf:` - Performance improvement

### Message Guidelines

**Length**: 50-72 characters maximum for the description

**Style**:
- Use imperative mood: "add" not "added" or "adds"
- No period at the end
- Lowercase after colon
- Be specific but concise

**Good Examples**:
```
feat: add user authentication with JWT
fix: handle null values in data processor
refactor: simplify search algorithm to O(n log n)
docs: update API documentation for v2 endpoints
test: add unit tests for payment validation
chore: update dependencies to latest versions
style: format code with prettier
perf: optimize database queries with indexes
```

**Bad Examples**:
```
Update stuff                    # Too vague
Added new feature for users.    # Past tense, period, too generic
Fix bug                         # Not descriptive enough
feat: Added the ability to...   # Past tense
FEAT: Add feature              # Wrong case
```

### Multi-File Changes

If changes span multiple concerns, pick the **primary** change:

**Priority order:**
1. Breaking changes or major features
2. Bug fixes
3. Refactoring
4. Everything else

**Examples**:
- Fix + tests → `fix: handle edge case in validator` (mention fix, not tests)
- Feature + docs → `feat: add export functionality` (mention feature, not docs)
- Refactor + tests + docs → `refactor: extract common utilities`

### Scope (Optional)

For larger projects, add scope in parentheses:

```
feat(auth): add password reset flow
fix(api): handle timeout errors
docs(readme): add installation instructions
```

## Step 4: Generate Message

### Analysis Process

1. **Count files by type**:
   - How many files changed?
   - What file types? (`.py`, `.js`, `.md`, `.json`)
   - New, modified, or deleted?

2. **Identify primary change**:
   - What's the main purpose of these changes?
   - What category does it fit?

3. **Extract key action**:
   - What verb describes the change? (add, fix, update, remove, improve, etc.)
   - What component/feature is affected?

4. **Craft concise description**:
   - Combine type + action + component
   - Keep under 72 characters
   - Use imperative mood

### Output Format

Provide the generated message in a code block for easy copying:

```
Generated commit message:

```
<type>: <description>
```

**Analysis:**
- Files changed: <count>
- Primary type: <type>
- Key changes: <brief summary>

**Alternative messages** (if applicable):
- `<alternative 1>`
- `<alternative 2>`
```

### Example Outputs

**Scenario 1**: Added new authentication module
```
Generated commit message:

```
feat: add JWT authentication module
```

**Analysis:**
- Files changed: 3 (auth.py, middleware.py, test_auth.py)
- Primary type: Feature
- Key changes: New authentication system with JWT tokens
```

**Scenario 2**: Fixed null pointer bug
```
Generated commit message:

```
fix: handle null values in user profile
```

**Analysis:**
- Files changed: 2 (profile.js, profile.test.js)
- Primary type: Bug fix
- Key changes: Added null checks to prevent crashes
```

**Scenario 3**: Updated dependencies
```
Generated commit message:

```
chore: update dependencies to latest versions
```

**Analysis:**
- Files changed: 2 (package.json, package-lock.json)
- Primary type: Maintenance
- Key changes: Bumped React, Jest, and ESLint versions
```

## Step 5: Handle Edge Cases

### No Changes Detected
```
No changes detected. Stage your changes first:
  git add <files>

Or check what's changed:
  git status
```

### Too Many Unrelated Changes
```
Warning: Changes span multiple concerns:
- 5 feature additions
- 3 bug fixes
- Documentation updates

Recommendation: Split into separate commits:
  git add <feature-files>
  git commit -m "feat: add user management"

  git add <fix-files>
  git commit -m "fix: handle validation errors"
```

### Large Refactoring
```
Generated commit message:

```
refactor: restructure data processing pipeline
```

**Note:** Large refactoring detected (500+ lines changed)
Consider adding a longer description:

```
refactor: restructure data processing pipeline

- Extract validation into separate module
- Simplify error handling flow
- Improve performance by 40%
```
```

## Usage Examples

### Basic Usage
```bash
# Stage your changes
git add .

# Run the command to get a message
/git-msg

# Copy the generated message and commit
git commit -m "feat: add user authentication"
```

### With Git Workflow
```bash
# After making changes
git status

# Stage specific files
git add src/auth.py tests/test_auth.py

# Generate message
/git-msg

# Commit with generated message
git commit -m "<generated-message>"
```

## Integration with /git-show

This command complements `/git-show`:
- **`/git-msg`**: Generate message for **current changes** (before commit)
- **`/git-show`**: Review **completed commits** (after commit)

**Workflow**:
1. Make changes
2. Use `/git-msg` to generate commit message
3. Commit with generated message
4. Use `/git-show` to review the commit

## Implementation Notes

### Bash Commands to Execute

```bash
# Step 1: Check what's staged
STAGED=$(git diff --cached --name-status)
UNSTAGED=$(git diff --name-status)

# Step 2: Get content
if [ -n "$STAGED" ]; then
    git diff --cached --stat
    git diff --cached
else
    git diff --stat
    git diff
fi

# Step 3: Analyze and generate message
# (Use pattern matching and heuristics described above)
```

### Priority Rules

1. If only documentation changed → `docs:`
2. If only test files changed → `test:`
3. If only config/dependencies changed → `chore:`
4. If bug fix in commit context → `fix:`
5. If new functionality → `feat:`
6. Otherwise → Analyze diff content

### Quality Checks

Before presenting message:
- ✅ Length < 72 characters
- ✅ Imperative mood
- ✅ Lowercase after prefix
- ✅ No period at end
- ✅ Specific and descriptive
