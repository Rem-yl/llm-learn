---
description: Review git commits or branch differences and highlight important changes
allowedTools: [Bash, Read, Grep]
---

# Git Show Command

Review git commits or workspace changes and provide detailed analysis with important highlights.

## Usage

- `/git-show` - Review the most recent commit
- `/git-show <commit-id>` - Review a specific commit (e.g., `/git-show abc1234`)
- `/git-show main` - Review current workspace/branch vs main branch
- `/git-show <branch>` - Review current workspace/branch vs any branch

## Step 1: Determine Review Mode

Parse the argument to determine what to review:

### Mode Detection Logic

```bash
# If no argument provided: review latest commit (HEAD)
# If argument looks like commit hash: verify and review that commit
# If argument is a branch name: compare current state vs that branch
```

**Commands to execute:**

```bash
# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

# If argument provided, determine if it's a commit or branch
if [[ -n "$ARG" ]]; then
    # Try as commit first
    if git cat-file -e "$ARG" 2>/dev/null; then
        MODE="commit"
        TARGET="$ARG"
    # Try as branch
    elif git rev-parse --verify "$ARG" 2>/dev/null; then
        MODE="branch"
        TARGET="$ARG"
    else
        echo "Error: '$ARG' is not a valid commit or branch"
        exit 1
    fi
else
    MODE="latest_commit"
    TARGET="HEAD"
fi
```

## Step 2: Gather Information Based on Mode

### For Commit Review (MODE=commit or MODE=latest_commit)

```bash
# Get commit metadata
git log -1 --format="Hash: %H%nAuthor: %an <%ae>%nDate: %ad%nSubject: %s%n%nBody:%n%b" "$TARGET"

# Get files changed with statistics
git show --stat --format="" "$TARGET"

# Get detailed diff
git show "$TARGET"

# Count changes
git show --numstat "$TARGET"
```

### For Branch Comparison (MODE=branch)

```bash
# Show what branch we're comparing
echo "Comparing: $CURRENT_BRANCH vs $TARGET"

# Get list of changed files
git diff --name-status "$TARGET"...HEAD

# Get commit log if on a branch
git log "$TARGET"..HEAD --oneline

# Get detailed diff
git diff "$TARGET"...HEAD

# Get statistics
git diff --numstat "$TARGET"...HEAD

# Also check for uncommitted changes
git status --porcelain
```

## Step 3: Analyze Changes

### Pattern Detection

Analyze the diff to identify important changes:

#### 1. Breaking Changes 🔴

Look for patterns indicating breaking changes:
- **Function signature changes**: Changed parameters or return types
- **Removed functions/methods**: Deleted function definitions
- **API endpoint changes**: Modified routes or HTTP methods
- **Configuration changes**: Changed config keys or formats
- **Database schema**: ALTER TABLE, DROP COLUMN, etc.
- **Removed exports**: Deleted `export`, `module.exports`, or public methods

**Search patterns:**
- `^-.*function|def|func|class|export`
- Parameter list changes in function definitions
- `ALTER TABLE|DROP COLUMN|MODIFY COLUMN`

#### 2. Security-Critical Changes 🔴

Look for security-sensitive code:
- **Authentication/Authorization**: Login, auth, permission checks
- **SQL queries**: String concatenation in queries (injection risk)
- **Input validation**: Missing sanitization
- **File operations**: File reads/writes without validation
- **Credentials**: API keys, passwords, tokens in code
- **CORS/Security headers**: Security configuration changes

**Search patterns:**
- `password|secret|api_key|token|credential`
- SQL string concatenation: `"SELECT.*" +|f"SELECT|${.*SELECT`
- `authenticate|authorize|permission|role`
- `open(|readFile|writeFile|fs\.`

#### 3. Performance Impact 🟡

Look for performance-critical changes:
- **Nested loops**: `O(n²)` complexity
- **Algorithm changes**: Different sorting/searching methods
- **Database queries**: N+1 queries, missing indexes
- **Synchronous operations**: Blocking calls in loops
- **Large data operations**: Loading entire files/datasets

**Search patterns:**
- Nested `for|while` loops
- `SELECT.*WHERE.*IN \(SELECT` (subqueries)
- Removed caching: `-.*cache|memoize`
- `sync|await` in loops

#### 4. Bug Fixes 🟢

Look for bug fix indicators:
- **Null checks added**: `if.*null|undefined|None`
- **Error handling**: New `try/catch` or error checks
- **Edge case handling**: Boundary checks
- **Logic corrections**: Operator changes (`==` to `===`, `>` to `>=`)

**Search patterns:**
- `\+ if.*null|\+ if.*undefined|\+ if.*None`
- `\+ try:|\+ try {`
- `\+ if len\(|if.*\.length`

#### 5. New Features 🟡

Look for new functionality:
- **New functions/classes**: Added definitions
- **New API endpoints**: New routes
- **New configuration**: New config options
- **New dependencies**: package.json, requirements.txt changes

**Search patterns:**
- `^\+.*function|def|func|class`
- `^\+.*@app\.route|@router|app\.get|app\.post`
- Changes in `package\.json|requirements\.txt|go\.mod`

#### 6. Test Changes 🟢

Look for test modifications:
- **New tests**: Added test functions
- **Modified tests**: Changed assertions
- **Removed tests**: Deleted test cases

**Search patterns:**
- `test_|spec\.|\.test\.|describe\(|it\(`
- `assert|expect\(|should`

#### 7. Dependency Changes 🟡

Check for dependency updates:
- `package.json`
- `requirements.txt`
- `go.mod`
- `Gemfile`
- `build.gradle`

## Step 4: Generate Review Output

### Output Structure

```markdown
# Git Show: [Mode Description]

## Overview

- **Reviewing**: [commit hash OR branch comparison]
- **Author**: [name] <email> (if commit)
- **Date**: [date] (if commit)
- **Commit Message**: [message] (if commit)
- **Files Changed**: [count]
- **Lines Added**: +[count]
- **Lines Removed**: -[count]

---

## 🔴 Critical Changes (Must Review Carefully)

[Only show this section if critical changes found]

### Security Concerns

- **[file.py:123]**: Hardcoded API key detected
  ```diff
  + api_key = "sk-1234567890abcdef"
  ```
  **Issue**: Credentials should not be in code
  **Recommendation**: Use environment variables

- **[auth.js:45]**: SQL injection vulnerability
  ```diff
  - const query = `SELECT * FROM users WHERE id = ${userId}`;
  + const query = "SELECT * FROM users WHERE id = " + userId;
  ```
  **Issue**: Unsafe string concatenation in SQL
  **Recommendation**: Use parameterized queries

### Breaking Changes

- **[api.py:67]**: Function signature changed
  ```diff
  - def process_data(data):
  + def process_data(data, options=None):
  ```
  **Impact**: Existing callers may break
  **Recommendation**: Add deprecation notice or provide migration guide

### Performance Regressions

- **[utils.js:89]**: Introduced O(n²) complexity
  ```diff
  + for (let i = 0; i < items.length; i++) {
  +   for (let j = 0; j < items.length; j++) {
  ```
  **Issue**: Nested loops on potentially large dataset
  **Recommendation**: Consider using a Set or Map for O(n) lookup

---

## 🟡 Important Changes

[Only show this section if important changes found]

### API Changes

- **[routes.py:34]**: New endpoint added
  ```diff
  + @app.route('/api/v2/users', methods=['POST'])
  ```

### Algorithm Changes

- **[search.py:12]**: Switched from linear to binary search
  ```diff
  - for item in items:
  + left, right = 0, len(items) - 1
  + while left <= right:
  ```
  **Impact**: Improved performance from O(n) to O(log n)

### Database Changes

- **[schema.sql:5]**: Added new column
  ```diff
  + ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
  ```
  **Note**: Requires database migration

### Dependency Changes

- **[package.json:15]**: Updated React version
  ```diff
  - "react": "^17.0.0"
  + "react": "^18.2.0"
  ```
  **Note**: Check for breaking changes in React 18

---

## 🟢 Other Notable Changes

### New Features

- **[feature.py:23]**: Added user export functionality
- **[api.js:67]**: Implemented rate limiting

### Bug Fixes

- **[utils.py:45]**: Fixed null pointer exception
  ```diff
  + if data is not None:
      process(data)
  ```

- **[validation.js:12]**: Fixed edge case for empty arrays
  ```diff
  + if (arr.length === 0) return [];
  ```

### Code Quality Improvements

- **[helper.py:10]**: Removed dead code
- **[format.js:34]**: Improved variable naming
- **[doc.md:1]**: Updated documentation

### Test Changes

- **[test_api.py:56]**: Added tests for new endpoint
- **[user.test.js:23]**: Updated assertions for new behavior

---

## Detailed File Analysis

[For each significant file:]

### 📄 src/auth/login.py (+45, -12)

**Change Type**: Security Enhancement
**Impact**: High

**What Changed**:
- Added rate limiting to login endpoint
- Improved password hashing algorithm
- Added account lockout after failed attempts

**Potential Issues**:
- ⚠️ Account lockout may affect legitimate users if threshold too low
- Existing password hashes need migration

**Suggestions**:
- Consider adding CAPTCHA for repeated failures
- Provide password migration script
- Add monitoring for lockout events

---

### 📄 src/utils/data_processor.js (+23, -67)

**Change Type**: Refactoring
**Impact**: Medium

**What Changed**:
- Simplified data processing logic
- Removed duplicate code
- Improved error handling

**Potential Issues**:
- None identified

**Suggestions**:
- Consider adding performance tests
- Ensure edge cases are covered in tests

---

## Code Quality Assessment

- **Style Consistency**: ✅ Follows project conventions
- **Error Handling**: ✅ Proper try/catch blocks added
- **Test Coverage**: ⚠️ Missing tests for new functions
- **Documentation**: ✅ Docstrings updated
- **Performance**: ✅ No performance concerns
- **Security**: ⚠️ See security concerns above

---

## Summary Statistics

### Files Changed by Type
- Python: 5 files
- JavaScript: 3 files
- SQL: 1 file
- Markdown: 2 files

### Change Distribution
- New features: 40%
- Bug fixes: 25%
- Refactoring: 20%
- Tests: 10%
- Documentation: 5%

---

## Recommendations

1. **Critical**: Fix SQL injection vulnerability in auth.js:45
2. **Important**: Add tests for new user export feature
3. **Important**: Document breaking changes in API
4. **Nice to have**: Add performance benchmarks for data processor
5. **Nice to have**: Update migration guide for React 18 upgrade

---

## Questions for Author

1. Was the password hashing migration tested with production data?
2. What is the account lockout threshold and duration?
3. Are there plans to add tests for the new export functionality?
4. Has the performance impact of the nested loops been measured?

---

## Overall Assessment

**Recommendation**: ⚠️ **Request Changes**

While this change includes valuable improvements, there are critical security and breaking change issues that should be addressed before merging.

**Key Concerns**:
- SQL injection vulnerability must be fixed
- Breaking changes need documentation
- Missing test coverage for new features

**Positive Aspects**:
- Good error handling improvements
- Performance optimization in search algorithm
- Clean code refactoring
```

## Implementation Notes

### For Each Review Mode

**Latest Commit (`/git-show`):**
- Review HEAD commit
- Show full commit metadata
- Analyze all changes in that commit

**Specific Commit (`/git-show abc1234`):**
- Verify commit exists
- Show commit metadata
- Analyze changes in that specific commit

**Branch Comparison (`/git-show main`):**
- Compare current state vs target branch
- Show all commits between branches
- Analyze cumulative diff
- Flag uncommitted changes if present

### Priority Rules

Show sections in order of priority:
1. 🔴 Critical (security, breaking, performance regression)
2. 🟡 Important (API, algorithms, dependencies, database)
3. 🟢 Notable (features, bugs, quality, tests, docs)

Only show sections that have content (skip empty sections).

### Analysis Depth

For each file:
- Small changes (< 50 lines): Show inline analysis
- Medium changes (50-200 lines): Summarize key points
- Large changes (> 200 lines): High-level overview with critical issues only

### File Reference Format

Always use `file:line` format for specific references:
- `auth.py:45` - Specific line
- `api.js:67-89` - Line range
- `schema.sql:12` - Database changes

This allows users to quickly navigate to issues.
