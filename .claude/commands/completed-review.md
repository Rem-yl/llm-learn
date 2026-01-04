---
description: Review code changes for bugs, documentation, code quality, and performance
allowedTools: [Bash, Read, Grep, Glob]
---

# Code Completion Review

Please perform a thorough review of the recent changes in this repository:

## 1. Identify Changed Files
- Use git to find files that have changed (staged, unstaged, or recent commits)
- If the repository has a main branch, compare against it
- Handle both new repositories and established ones
- Commands: `git status`, `git diff`, `git diff --cached`

## 2. Bug Analysis
- Null/undefined handling and edge cases
- Logic errors and incorrect conditionals
- Error handling presence and appropriateness
- Race conditions or concurrency issues
- Security vulnerabilities (injection, XSS, auth issues)
- Off-by-one errors and boundary conditions

## 3. Code Quality
- **Style Consistency**: formatting, naming conventions, indentation
- **Code Duplication**: repeated logic that should be extracted
- **Complexity**: overly complex functions needing refactoring
- **Dead Code**: unused variables, imports, or commented code
- **Debug Artifacts**: console.logs, debugger statements, temporary code
- **Magic Numbers**: hardcoded values that should be constants
- **Function Length**: functions that are too long and need splitting

## 4. Performance
- **Algorithm Efficiency**: inefficient loops or O(n²) operations
- **Unnecessary Computations**: redundant calculations or re-renders
- **Memory Management**: potential memory leaks or large allocations
- **Database Queries**: N+1 queries or missing indexes
- **File Operations**: large file reads without streaming
- **Bundle Size**: unnecessarily large imports or dependencies

## 5. Documentation Completeness

### For New Functions/Methods:
- **Comprehensive Docstrings**: All new functions must have detailed documentation
  - Purpose: What the function does
  - Parameters: Type, description, and constraints for each parameter
  - Returns: Return type and description
  - Raises/Throws: What exceptions can be raised
  - Examples: Usage examples showing common cases
  - **Python-specific**: Include doctest examples (simple unittest code in docstrings)
  - **JavaScript/TypeScript**: JSDoc with @example tags
  - **Complex functions**: Additional explanation of algorithm or approach

### For Modified Functions:
- **Change Documentation**: If a function has been significantly changed:
  - Add inline comments explaining what changed and why
  - Update existing docstrings to reflect new behavior
  - Document breaking changes or behavior differences
  - Add TODO/FIXME if changes are incomplete

### General Documentation:
- README updated for new features
- API documentation reflects changes
- Configuration changes documented
- Breaking changes highlighted

### Review Checklist:
- [ ] New complex functions have comprehensive docstrings with examples
- [ ] Python functions include doctest examples where appropriate
- [ ] Modified functions have comments explaining the changes
- [ ] All parameters and return values documented
- [ ] Edge cases and exceptions documented
- [ ] Public API changes reflected in docs

## 6. Summary Report

Provide a structured summary:

### Files Reviewed
List all files analyzed with line counts changed

### Critical Issues
Bugs or security concerns that MUST be fixed before merging

### Code Quality Issues
Improvements needed for maintainability

### Performance Concerns
Optimization opportunities identified

### Documentation Gaps
Missing or incomplete documentation that needs attention

### Recommendations
Prioritized action items with specific file locations
