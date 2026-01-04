---
description: Create test code for new functions, verify they run, and update test documentation
allowedTools: [Bash, Read, Write, Edit, Grep, Glob]
---

# Add Test Command

Create comprehensive test code for new or modified functions, verify tests are runnable, and maintain test documentation.

## Phase 1: Identify Functions to Test

### Detection Methods
- Analyze `git diff` to find new or modified functions
- Accept user-specified functions/files
- Detect programming language (Python, JavaScript/TypeScript, Go, Java, etc.)
- Identify existing test framework (pytest, jest, mocha, JUnit, go test, etc.)
- Analyze function signatures, parameters, return values, docstrings

### Check for Significant Logic Changes
Determine if function logic has been significantly changed by examining git diff:

**Indicators of significant logic change:**
- Algorithm replacement (different sorting/searching method)
- Loop structure substantially modified (nested → single, iteration approach changed)
- Data structure changes (list → set, array → map)
- Time complexity changes (O(n²) → O(n))
- New recursive approach or removed recursion
- Database query pattern changes
- External API call changes

## Phase 2: Generate Test Code

### Primary Goal: Verify Functional Correctness

The main objective is to verify the function works correctly and produces expected results.

### Step 1: Always Generate Functional Tests

Generate tests to verify **correctness and expected behavior**:

#### Python (pytest)

**Test File Naming**: `test_<module>.py`

```python
# test_module.py
import pytest
from module import function_to_test

class TestFunctionToTest:
    """Verify function_to_test works correctly"""

    def test_correct_output_for_valid_input(self):
        """Verify function produces correct result"""
        result = function_to_test([1, 2, 3])
        assert result == [1, 4, 9]  # Expected correct output

    @pytest.mark.parametrize("input,expected", [
        ([1, 2], [1, 4]),
        ([0], [0]),
        ([5, 5], [25, 25]),
    ])
    def test_correctness_various_inputs(self, input, expected):
        """Verify correct behavior across different inputs"""
        assert function_to_test(input) == expected

    def test_empty_input_handled_correctly(self):
        """Verify edge case: empty input"""
        assert function_to_test([]) == []

    def test_none_input_handled_correctly(self):
        """Verify edge case: None input"""
        assert function_to_test(None) == []  # Or raises error

    def test_boundary_values(self):
        """Verify boundary conditions"""
        assert function_to_test([sys.maxsize]) is not None
        assert function_to_test([-sys.maxsize]) is not None

    def test_invalid_input_raises_error(self):
        """Verify error handling works correctly"""
        with pytest.raises(TypeError):
            function_to_test("not a list")

    def test_side_effects_work_correctly(self):
        """Verify any side effects occur as expected"""
        # E.g., database updates, file writes, state changes
        result = function_to_test(data)
        assert database.count() == expected_count

    @pytest.fixture
    def setup_data(self):
        """Setup test data"""
        return {"key": "value"}
```

#### JavaScript/TypeScript (Jest)

**Test File Naming**: `<module>.test.js` or `<module>.spec.js`

```javascript
// module.test.js
import { functionToTest } from './module';

describe('functionToTest', () => {
  describe('Correctness Tests', () => {
    it('should produce correct output for valid input', () => {
      const result = functionToTest([1, 2, 3]);
      expect(result).toEqual([1, 4, 9]);
    });

    it.each([
      [[1, 2], [1, 4]],
      [[0], [0]],
      [[5, 5], [25, 25]],
    ])('should return correct result for %s', (input, expected) => {
      expect(functionToTest(input)).toEqual(expected);
    });

    it('should handle empty array correctly', () => {
      expect(functionToTest([])).toEqual([]);
    });

    it('should handle null correctly', () => {
      expect(functionToTest(null)).toEqual([]);
    });

    it('should handle undefined correctly', () => {
      expect(functionToTest(undefined)).toEqual([]);
    });

    it('should throw error for invalid input', () => {
      expect(() => functionToTest('invalid')).toThrow(TypeError);
    });

    it('should handle boundary values', () => {
      expect(functionToTest([Number.MAX_VALUE])).toBeDefined();
      expect(functionToTest([Number.MIN_VALUE])).toBeDefined();
    });
  });

  describe('Async operations', () => {
    it('should handle async execution correctly', async () => {
      const result = await asyncFunction();
      expect(result).toBeDefined();
    });
  });

  beforeEach(() => {
    // Setup before each test
  });

  afterEach(() => {
    // Cleanup after each test
    jest.clearAllMocks();
  });
});
```

#### Go (testing)

**Test File Naming**: `<file>_test.go`

```go
// module_test.go
package mypackage

import (
    "reflect"
    "testing"
)

func TestFunctionToTest(t *testing.T) {
    tests := []struct {
        name     string
        input    []int
        expected []int
        wantErr  bool
    }{
        {
            name:     "correct output for valid input",
            input:    []int{1, 2, 3},
            expected: []int{1, 4, 9},
            wantErr:  false,
        },
        {
            name:     "handles empty slice correctly",
            input:    []int{},
            expected: []int{},
            wantErr:  false,
        },
        {
            name:     "handles nil input correctly",
            input:    nil,
            expected: []int{},
            wantErr:  false,
        },
        {
            name:     "handles single element",
            input:    []int{5},
            expected: []int{25},
            wantErr:  false,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result, err := FunctionToTest(tt.input)

            if (err != nil) != tt.wantErr {
                t.Errorf("FunctionToTest() error = %v, wantErr %v", err, tt.wantErr)
                return
            }

            if !reflect.DeepEqual(result, tt.expected) {
                t.Errorf("FunctionToTest() = %v, want %v", result, tt.expected)
            }
        })
    }
}
```

#### Test Coverage Requirements (Always)

For each function, generate tests for:

1. **Correctness**: Function produces correct output
2. **Happy Path**: Normal execution with valid inputs
3. **Edge Cases**:
   - Empty inputs ([], "", null, undefined, 0)
   - Boundary values (min, max, first, last)
   - Single element
4. **Error Cases**:
   - Invalid inputs
   - Type mismatches
   - Null/undefined handling
   - Appropriate exceptions raised
5. **Side Effects**: State changes, database updates work as intended

### Step 2: Conditionally Generate Performance Tests

**Only generate performance tests if function logic has been significantly changed.**

#### Detection Criteria

Check git diff and only add performance tests if:
- ✅ Algorithm was replaced (different approach)
- ✅ Loop structure substantially modified
- ✅ Data structure changed (affects complexity)
- ✅ Time complexity improved or changed
- ✅ Recursive approach added/removed

**Do NOT generate performance tests for:**
- ❌ New functions (no baseline to compare)
- ❌ Minor bug fixes
- ❌ Code style refactoring
- ❌ Simple getter/setter functions
- ❌ Functions without logic changes

#### Python (pytest-benchmark)

**Only if logic significantly changed:**

```python
# test_performance.py
import pytest

class TestFunctionPerformance:
    """Performance tests - only because function logic was significantly changed"""

    def test_performance_baseline(self, benchmark):
        """Establish performance baseline after logic change"""
        data = list(range(1000))
        result = benchmark(function_to_test, data)
        assert result is not None

    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_scales_as_expected(self, size):
        """Verify new algorithm scales correctly (e.g., O(n) not O(n²))"""
        import time
        data = list(range(size))

        start = time.perf_counter()
        result = function_to_test(data)
        duration = time.perf_counter() - start

        # Verify expected complexity after optimization
        assert duration < size * 0.0001  # Adjust based on expected O(n)

    def test_no_performance_regression(self):
        """Verify change didn't make performance worse"""
        data = list(range(1000))

        import time
        start = time.perf_counter()
        result = function_to_test(data)
        duration = time.perf_counter() - start

        # Should be faster than old implementation baseline
        assert duration < 0.1  # Based on old implementation

    def test_memory_usage_acceptable(self):
        """Verify memory usage is reasonable"""
        import tracemalloc

        tracemalloc.start()
        result = function_to_test(list(range(10000)))
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should not use excessive memory
        assert peak < 10 * 1024 * 1024  # < 10MB
```

#### JavaScript (Jest)

**Only if logic significantly changed:**

```javascript
// performance.test.js
describe('Performance Tests - After Significant Logic Change', () => {
  it('should be faster than previous implementation', () => {
    const data = Array.from({ length: 1000 }, (_, i) => i);
    const start = performance.now();
    functionToTest(data);
    const duration = performance.now() - start;

    // Verify improved performance vs old implementation
    expect(duration).toBeLessThan(50); // Old was 200ms
  });

  it('should scale linearly (O(n)) after optimization', () => {
    const sizes = [100, 1000, 10000];
    const timings = sizes.map(size => {
      const data = Array.from({ length: size }, (_, i) => i);
      const start = performance.now();
      functionToTest(data);
      return performance.now() - start;
    });

    // Verify linear scaling (not quadratic)
    const ratio1 = timings[1] / timings[0]; // Should be ~10x
    const ratio2 = timings[2] / timings[1]; // Should be ~10x

    expect(ratio1).toBeLessThan(15); // Allow some overhead
    expect(ratio2).toBeLessThan(15);
  });

  it('should not leak memory', () => {
    const iterations = 100;
    const initialMemory = process.memoryUsage().heapUsed;

    for (let i = 0; i < iterations; i++) {
      functionToTest(Array.from({ length: 1000 }, (_, i) => i));
    }

    if (global.gc) global.gc(); // Requires node --expose-gc
    const finalMemory = process.memoryUsage().heapUsed;
    const memoryGrowth = finalMemory - initialMemory;

    expect(memoryGrowth).toBeLessThan(5 * 1024 * 1024); // < 5MB
  });
});
```

#### Go (benchmarks)

**Only if logic significantly changed:**

```go
// module_test.go

// BenchmarkFunctionToTest_AfterOptimization establishes new baseline
func BenchmarkFunctionToTest_AfterOptimization(b *testing.B) {
    data := make([]int, 1000)
    for i := range data {
        data[i] = i
    }
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        FunctionToTest(data)
    }
}

// BenchmarkFunctionToTest_Scaling verifies expected complexity
func BenchmarkFunctionToTest_Scaling(b *testing.B) {
    sizes := []int{100, 1000, 10000}

    for _, size := range sizes {
        b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
            data := make([]int, size)
            for i := range data {
                data[i] = i
            }
            b.ResetTimer()

            for i := 0; i < b.N; i++ {
                FunctionToTest(data)
            }
        })
    }
}

// BenchmarkFunctionToTest_Memory checks allocations
func BenchmarkFunctionToTest_Memory(b *testing.B) {
    b.ReportAllocs()
    data := make([]int, 1000)
    for i := range data {
        data[i] = i
    }

    for i := 0; i < b.N; i++ {
        FunctionToTest(data)
    }
}
```

### Decision Tree for Test Generation

```
Function detected (new or modified)
│
├─ STEP 1: Generate Functional Tests (ALWAYS)
│  └─ Verify: correctness, edge cases, errors, side effects
│
└─ STEP 2: Check git diff - was logic significantly changed?
   │
   ├─ YES: Algorithm/loop/complexity changed
   │  └─ Generate Performance Tests
   │     └─ Verify: speed, scalability, no regression, memory
   │
   └─ NO: New function or minor changes
      └─ Skip performance tests
```

## Phase 3: Verify Tests are Runnable

### Execution Steps

1. **Detect test framework and commands**
   - Python: `pytest`, `python -m pytest`
   - JavaScript: `npm test`, `yarn test`, `jest`
   - Go: `go test`
   - Look for config files: `pytest.ini`, `jest.config.js`, etc.

2. **Run functional tests first**
   - Execute with appropriate test runner
   - Capture output (stdout, stderr)
   - Check exit code

3. **Handle errors**
   - **Syntax errors**: Fix test code and retry
   - **Import errors**: Check dependencies, fix imports, verify paths
   - **Test failures**: Distinguish between:
     - Test code issues (fix immediately)
     - Implementation bugs (report to user, don't mark complete)

4. **Verify success criteria**
   - At least one test passes (confirms setup works)
   - No syntax or import errors
   - Tests execute completely without crashes

5. **Run performance tests separately** (if generated)
   - May take longer to execute
   - Establish baseline metrics
   - Verify realistic thresholds
   - Save baselines for future comparison

### Commands to Execute

#### Python
```bash
# Run functional tests
pytest tests/test_module.py -v

# If tests fail, run with more detail
pytest tests/test_module.py -vv --tb=long

# Run performance tests separately
pytest tests/test_performance.py --benchmark-only
```

#### JavaScript
```bash
# Run functional tests
npm test -- module.test.js

# Run with verbose output if failures
npm test -- module.test.js --verbose

# Run performance tests
npm test -- performance.test.js
```

#### Go
```bash
# Run functional tests
go test -v -run TestFunctionToTest

# Run benchmarks separately
go test -bench=. -benchmem
```

### Error Handling

If tests fail to run:
1. Check and fix syntax errors
2. Verify imports and dependencies
3. Ensure test framework is installed
4. Fix file paths and naming conventions
5. Retry until at least one test passes

If implementation has bugs:
- Report to user
- Do NOT mark test creation as complete
- Suggest fixes or ask user to fix implementation first

## Phase 4: Create/Update test_run.md

After generating and verifying tests, create or update `test_run.md` with comprehensive documentation.

### test_run.md Structure

```markdown
# Test Running Guide

## Quick Start

### Run All Tests
[Command to run entire test suite for this project]

### Run Specific Test File
[Command to run one test file]

### Run with Coverage
[Command to see test coverage]

## Test Frameworks Used

- **Python**: pytest v[version]
  - Config: `pytest.ini` or `pyproject.toml`
  - Performance: pytest-benchmark

- **JavaScript/TypeScript**: Jest v[version]
  - Config: `jest.config.js`

- **Go**: testing (built-in)
  - No additional config needed

## Running Tests

### Python (pytest)

\`\`\`bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_module.py

# Run specific test function
pytest tests/test_module.py::TestClass::test_function

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=module --cov-report=html tests/

# Run tests matching pattern
pytest -k "test_pattern"

# Run failed tests from last run
pytest --lf

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
\`\`\`

### JavaScript/TypeScript (Jest)

\`\`\`bash
# Run all tests
npm test

# Run specific test file
npm test -- module.test.js

# Run tests matching pattern
npm test -- --testNamePattern="pattern"

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch

# Update snapshots
npm test -- -u

# Run with verbose output
npm test -- --verbose
\`\`\`

### Go (testing)

\`\`\`bash
# Run all tests
go test ./...

# Run tests in current package
go test

# Run specific test
go test -run TestFunctionName

# Run with verbose output
go test -v ./...

# Run with coverage
go test -cover ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run tests in parallel
go test -parallel 4 ./...
\`\`\`

## Performance Testing

### When to Run Performance Tests

Run performance tests when:
- Function logic has been significantly changed
- Before deploying optimizations to production
- Investigating performance issues
- Establishing performance baselines
- Validating expected time complexity

### Running Performance Tests

#### Python (pytest-benchmark)

\`\`\`bash
# Run performance tests only
pytest tests/test_performance.py --benchmark-only

# Run and compare with baseline
pytest tests/test_performance.py --benchmark-compare=baseline

# Save new baseline
pytest tests/test_performance.py --benchmark-save=baseline

# Generate histogram
pytest tests/test_performance.py --benchmark-histogram

# Run with specific number of rounds
pytest tests/test_performance.py --benchmark-min-rounds=10
\`\`\`

#### JavaScript (Jest with performance tests)

\`\`\`bash
# Run performance tests
npm test -- performance.test.js

# Run with custom timeout for slow tests
npm test -- --testTimeout=30000

# Run with Node.js garbage collection exposed
node --expose-gc node_modules/.bin/jest performance.test.js
\`\`\`

#### Go (benchmarks)

\`\`\`bash
# Run all benchmarks
go test -bench=. -benchmem

# Run specific benchmark
go test -bench=BenchmarkFunctionName -benchmem

# Run benchmark multiple times for accuracy
go test -bench=. -benchtime=10s

# Save benchmark results
go test -bench=. -benchmem > benchmark.txt

# Compare benchmarks (requires benchstat: go install golang.org/x/perf/cmd/benchstat@latest)
go test -bench=. -benchmem > new.txt
benchstat old.txt new.txt

# CPU profiling
go test -bench=. -cpuprofile=cpu.prof
go tool pprof cpu.prof

# Memory profiling
go test -bench=. -memprofile=mem.prof
go tool pprof mem.prof
\`\`\`

## Understanding Test Output

### Success Indicators

**Python (pytest):**
```
test_module.py::TestClass::test_function PASSED              [100%]
===================== 5 passed in 0.23s ======================
```

**JavaScript (Jest):**
```
 PASS  tests/module.test.js
  ✓ should handle valid input (3 ms)

Test Suites: 1 passed, 1 total
Tests:       5 passed, 5 total
```

**Go:**
```
=== RUN   TestFunctionName
--- PASS: TestFunctionName (0.00s)
PASS
ok      mypackage       0.234s
```

### Failure Indicators

**Python (pytest):**
```
test_module.py::test_function FAILED                         [100%]
E   AssertionError: assert 5 == 10
E   Expected: 10
E   Actual: 5
===================== 1 failed in 0.45s ======================
```

**JavaScript (Jest):**
```
 FAIL  tests/module.test.js
  ✕ should handle valid input (5 ms)

  Expected: 10
  Received: 5

Test Suites: 1 failed, 1 total
Tests:       1 failed, 5 total
```

**Go:**
```
--- FAIL: TestFunctionName (0.00s)
    function_test.go:15: FunctionName(5) = 5; want 10
FAIL
FAIL    mypackage       0.123s
```

### Performance Test Output

**Python (pytest-benchmark):**
```
Name                         Min        Max       Mean    StdDev
test_performance          1.23ms     1.45ms    1.30ms    0.05ms
```
- **Min/Max**: Fastest/slowest execution
- **Mean**: Average execution time (lower is better)
- **StdDev**: Consistency (lower is better)

**Go (benchmarks):**
```
BenchmarkFunctionName-8    50000    23456 ns/op    1024 B/op    10 allocs/op
```
- **50000**: Number of iterations
- **23456 ns/op**: Nanoseconds per operation (lower is better)
- **1024 B/op**: Bytes allocated per operation (lower is better)
- **10 allocs/op**: Number of allocations (lower is better)

## Test Types in This Project

### Functional Tests (Always Present)
- Test correctness and expected behavior
- Fast execution
- Run on every commit
- Location: `tests/` or `test_*.py/js/go`

### Performance Tests (When Logic Changed)
- Benchmark execution time and memory
- Validate scalability
- Detect regressions after optimizations
- Location: `tests/test_performance.py` or `performance.test.js`

## Troubleshooting

### Common Issues

**Import Errors**
```
ModuleNotFoundError: No module named 'mymodule'
```
**Solution**:
- Python: Install dependencies: `pip install -r requirements.txt`
- Check PYTHONPATH or use `pip install -e .`
- JavaScript: Run `npm install`

**Missing Test Dependencies**
```
Error: Cannot find module 'jest'
```
**Solution**:
- JavaScript: `npm install --save-dev jest`
- Python: `pip install pytest pytest-benchmark`

**Test Configuration Issues**
```
No tests found matching pattern
```
**Solution**:
- Check test file naming convention
- Verify test discovery settings
- Python: Check `pytest.ini` or `pyproject.toml`
- JavaScript: Check `jest.config.js`

**Performance Test Timeouts**
```
Test exceeded timeout of 5000ms
```
**Solution**:
- Increase timeout: `jest.setTimeout(30000)`
- Or optimize function performance

### Debugging Failing Tests

**Verbose Output:**
```bash
# Python
pytest -vv --tb=long

# JavaScript
npm test -- --verbose

# Go
go test -v
```

**Run Single Test:**
```bash
# Python
pytest tests/test_module.py::test_function -vv

# JavaScript
npm test -- --testNamePattern="specific test"

# Go
go test -run TestSpecificFunction -v
```

**Use Debugger:**
```bash
# Python
pytest --pdb  # Drops into debugger on failure

# JavaScript
node --inspect-brk node_modules/.bin/jest --runInBand

# Go (with delve)
dlv test -- -test.run TestFunctionName
```

### Setting Realistic Performance Thresholds

**Guidelines by dataset size:**
- **Small** (< 100 items): < 10ms
- **Medium** (< 10,000 items): < 100ms
- **Large** (< 1M items): < 1 second
- **Very large** (> 1M items): < 10 seconds

**Adjust based on:**
- Function complexity
- I/O operations (disk, network, database)
- Hardware capabilities
- Acceptable latency for use case

**Example thresholds:**
```python
# For O(n) algorithm processing 1000 items
assert duration < 0.01  # 10ms

# For O(n log n) sort of 10000 items
assert duration < 0.05  # 50ms

# For database query
assert duration < 0.5   # 500ms
```

## Coverage Goals

- **Functional test coverage**: > 80%
- **Critical paths**: 100%
- **Performance tests**: All functions with significant logic changes

**View coverage reports:**
```bash
# Python
pytest --cov=module --cov-report=html
open htmlcov/index.html

# JavaScript
npm test -- --coverage
open coverage/lcov-report/index.html

# Go
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

## Continuous Integration

### GitHub Actions Example
```yaml
- name: Run tests
  run: |
    pytest --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### GitLab CI Example
```yaml
test:
  script:
    - npm test -- --coverage
  coverage: '/Lines\\s*:\\s*(\\d+\\.\\d+)%/'
```

## Summary

- **Functional tests**: Always run to verify correctness
- **Performance tests**: Run when logic has significantly changed
- **Coverage**: Aim for > 80% overall, 100% for critical paths
- **CI/CD**: Run functional tests on every commit
```

### Documentation Update Process

After running `/add-test`:
1. ✅ Test files are created
2. ✅ Tests are verified to run successfully
3. ✅ `test_run.md` is created or updated
4. ✅ User knows exactly how to run and interpret tests

User receives:
- Working test code (functional + performance if needed)
- Clear documentation on running tests
- Understanding of test output
- Troubleshooting guidance
