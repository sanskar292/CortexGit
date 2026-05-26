# Contributing to CortexGit

Thank you for your interest in contributing to CortexGit! We welcome community contributions to help improve this persistent memory library for LLM agents.

---

## 🛠️ 1. Development Setup

To contribute code, set up a local development environment:

1. **Fork and Clone** the repository.
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or .\venv\Scripts\Activate.ps1 on Windows
   ```
3. **Install the package in editable mode** with development dependencies:
   ```bash
   pip install -e .[dev]
   ```

---

## 🧪 2. Running Tests

We use `pytest` to run our test suite. Ensure all tests pass before submitting changes:

```bash
pytest tests/ -v
```

---

## 🎨 3. Code Formatting

We use `black` to enforce consistent code style. Format your changes before committing:

```bash
black src/ tests/
```

---

## 🔍 4. Linting and Static Analysis

We use `ruff` for fast linting and code quality checks. Run the linter to verify your changes:

```bash
ruff check src/ tests/
```

---

## 🔀 5. Pull Request Process

Follow these steps to submit a change:

1. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Implement changes** and write corresponding unit tests under `tests/`.
3. **Verify locally**: Make sure that:
   - All tests pass (`pytest tests/ -v`)
   - Code is formatted (`black src/ tests/`)
   - Linter is clean (`ruff check src/ tests/`)
4. **Push your branch** to your fork and **open a Pull Request** against our `main` branch.

---

## 📝 6. Code Style Guidelines

- **Asynchronous Code**: The database interaction layer relies heavily on asynchronous SQLAlchemy and `aiosqlite`. Ensure all public APIs maintain `async`/`await` patterns where applicable.
- **Type Hints**: Use type hints on all public function and method signatures.
- **Documentation**: Maintain or update docstrings for any added or modified functions. Update documentation under the `docs/` folder if changing public API behavior.
- **Backward Compatibility**: Be mindful when modifying public methods on `CortexGit` (such as `log_event` or `get_context`) to avoid breaking existing integrations.
