# CortexGit — From Code to Library
## How to package, version, and distribute your working code

---

## What You Have

A working Python project with:
- Core memory system (event log, entity registry, conflict detection)
- Retrieval pipeline (semantic search, context assembly)
- LLM integration (snapshots, entity extraction)
- Full test suite

Now you make it installable.

---

## Step 1 — Project Structure Cleanup

Your current structure:
```
cortexgit/
  api/                    ← DELETE THIS (we're SDK now, not HTTP)
  core/
  llm/
  retrieval/
  db/
  schemas/
  tests/
  ARCHITECTURE.md
  PROGRESS.md
  requirements.txt
  pytest.ini
```

New structure:
```
cortexgit/                ← Root folder
  src/
    cortexgit/            ← Package folder (what gets installed)
      __init__.py         ← Exports: from cortexgit import CortexGit
      core/
        __init__.py
        memory.py         ← Main CortexGit class lives here
        event_log.py
        entity_registry.py
        conflict_detector.py
        write_back_gate.py
      retrieval/
        __init__.py
        context_assembler.py
        semantic_recall.py
        recency_filter.py
        entity_pull.py
      llm/
        __init__.py
        summarizer.py
        entity_extractor.py
        snapshot_trigger.py
      schemas/
        __init__.py
        snapshot_schema.json
        entity_extraction_schema.json
  tests/                  ← Tests stay at root
    conftest.py
    test_event_log.py
    test_entity_registry.py
    ...
  docs/
    ARCHITECTURE.md
    GETTING_STARTED.md
    API_REFERENCE.md
  .github/
    workflows/
      tests.yml           ← CI/CD
  setup.py                ← NEW
  pyproject.toml          ← NEW
  MANIFEST.in             ← NEW
  .gitignore
  README.md               ← Rewrite this
  LICENSE
  requirements.txt        ← Keep for development
  requirements-dev.txt    ← NEW (test deps)
```

---

## Step 2 — Create setup.py

This is what tells Python how to install your package.

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="cortexgit",
    version="0.1.0",
    description="Persistent memory for LLM agents. Event sourcing + semantic retrieval.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourname/cortexgit",
    license="MIT",
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    
    python_requires=">=3.10",
    
    install_requires=[
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",  # PostgreSQL adapter
        "asyncpg>=0.28.0",         # Async PostgreSQL adapter
        "pgvector>=0.1.0",
        "anthropic>=0.25.0",
        "openai>=1.0.0",
        "jsonschema>=4.0.0",
        "python-dotenv>=1.0.0",
        "aiosqlite>=0.19.0",       # Async SQLite driver
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    keywords="ai agents memory persistence event-sourcing semantic-search",
    project_urls={
        "Bug Reports": "https://github.com/yourname/cortexgit/issues",
        "Source": "https://github.com/yourname/cortexgit",
        "Documentation": "https://cortexgit.readthedocs.io",
    },
)
```

---

## Step 3 — Create pyproject.toml

Modern Python prefers this over setup.py, but you need both during transition.

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cortexgit"
version = "0.1.0"
description = "Persistent memory for LLM agents"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Your Name", email = "you@example.com"}]

dependencies = [
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "asyncpg>=0.28.0",
    "pgvector>=0.1.0",
    "anthropic>=0.25.0",
    "openai>=1.0.0",
    "jsonschema>=4.0.0",
    "python-dotenv>=1.0.0",
    "aiosqlite>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.urls]
Repository = "https://github.com/yourname/cortexgit"
Issues = "https://github.com/yourname/cortexgit/issues"

[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/cortexgit --cov-report=html"
```

---

## Step 4 — Update __init__.py

This is what users import.

```python
# src/cortexgit/__init__.py

from src.cortexgit.core.memory import CortexGit
from src.cortexgit.core.event_log import EventLog
from src.cortexgit.core.entity_registry import EntityRegistry

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "CortexGit",
    "EventLog",
    "EntityRegistry",
]
```

Users just do:
```python
from cortexgit import CortexGit

memory = CortexGit()
```

---

## Step 5 — Create MANIFEST.in

Tells setuptools what non-Python files to include.

```
# MANIFEST.in
include README.md
include LICENSE
include ARCHITECTURE.md
recursive-include src/cortexgit/schemas *.json
recursive-include tests *.py
```

---

## Step 6 — Create requirements-dev.txt

For developers who want to contribute or work locally.

```
# requirements-dev.txt
-e .[dev]

# Add these for local development
pytest-watch>=4.2.0
ipython>=8.0.0
black>=23.0.0
ruff>=0.1.0
```

Then developers just run:
```bash
pip install -r requirements-dev.txt
```

---

## Step 7 — Update README.md

Make it marketing + installation.

```markdown
# CortexGit

Persistent memory for LLM agents. Event sourcing + semantic retrieval.

## The Problem

LLM agents are stateless. They forget context between sessions. They can't coordinate without explicit message passing. They have no audit trail.

## The Solution

CortexGit is an in-process memory system. Write events, retrieve context, persist facts. Works with any LLM, any agent framework.

## Installation

```bash
pip install cortexgit
```

## Quick Start

```python
from cortexgit import CortexGit
from anthropic import Anthropic

# Initialize memory (creates local SQLite database)
memory = CortexGit()
client = Anthropic()

def my_agent(user_query):
    # Retrieve relevant context from memory
    context = memory.get_context(
        goal=user_query,
        budget_tokens=4000
    )
    
    # Call Claude with context
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=f"You are a helpful agent. Memory: {context}",
        messages=[{"role": "user", "content": user_query}]
    )
    
    # Remember what happened
    memory.log_event("interaction", {
        "query": user_query,
        "response": response.content[0].text
    })
    
    return response.content[0].text

# Use it
print(my_agent("What is 2+2?"))
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Getting Started](docs/GETTING_STARTED.md)

## Features

- ✅ Append-only event log (source of truth)
- ✅ Persistent entity registry with conflict detection
- ✅ Automatic snapshot generation
- ✅ Semantic retrieval over compressed memory
- ✅ Works with any LLM (Claude, GPT, local models)
- ✅ Single import, no server needed

## License

MIT
```

---

## Step 8 — Create .gitignore

```
# .gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Environment variables
.env
.env.local

# Databases
*.db
*.sqlite
*.sqlite3

# Generated files
cortexgit.egg-info/
```

---

## Step 9 — GitHub Actions (CI/CD)

Make tests run automatically on every push.

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      
      - name: Lint with ruff
        run: ruff check src/ tests/
      
      - name: Format check with black
        run: black --check src/ tests/
      
      - name: Run tests
        run: pytest tests/ --cov=src/cortexgit
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Step 10 — Build and Test Locally

Before uploading to PyPI, test installation locally:

```bash
# Clean any old builds
rm -rf build/ dist/ *.egg-info/

# Build the distribution files
python -m build

# This creates:
# - dist/cortexgit-0.1.0.tar.gz (source distribution)
# - dist/cortexgit-0.1.0-py3-none-any.whl (wheel)

# Test installation in a clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/cortexgit-0.1.0-py3-none-any.whl

# Verify it works
python -c "from cortexgit import CortexGit; print('Success!')"
```

---

## Step 11 — Upload to PyPI

First, create accounts:

1. **PyPI** — https://pypi.org/account/register/
2. **TestPyPI** (optional, for testing) — https://test.pypi.org/account/register/

Then:

```bash
# Install twine (upload tool)
pip install twine

# Create ~/.pypirc with your credentials
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your PyPI token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your TestPyPI token
EOF

# Set strict permissions
chmod 600 ~/.pypirc

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test it
pip install -i https://test.pypi.org/simple/ cortexgit

# If all good, upload to real PyPI
twine upload dist/*
```

Now anyone can do:
```bash
pip install cortexgit
```

---

## Step 12 — Version Management

For future releases, follow semantic versioning:

```
0.1.0  ← Current (alpha release)
0.2.0  ← Next feature release
0.3.0  ← Bugfixes accumulate
1.0.0  ← Stable/production ready
```

Update version in:
1. `setup.py` — `version="0.2.0"`
2. `pyproject.toml` — `version = "0.2.0"`
3. `src/cortexgit/__init__.py` — `__version__ = "0.2.0"`
4. Tag in git: `git tag v0.2.0`

Then rebuild and upload:
```bash
python -m build
twine upload dist/*
```

---

## Step 13 — Documentation (Optional but Recommended)

Create `docs/` folder with:

```
docs/
  GETTING_STARTED.md     ← Installation + first example
  API_REFERENCE.md       ← Every public method
  ARCHITECTURE.md        ← How it works internally
  EXAMPLES.md            ← Real-world use cases
  TROUBLESHOOTING.md     ← Common issues
```

Then push to ReadTheDocs for free hosting:

1. Go to https://readthedocs.org
2. Import your GitHub repo
3. It auto-builds docs on every push

---

## Final Checklist

Before you publish v0.1.0:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black src/ tests/`
- [ ] Code is linted: `ruff check src/ tests/`
- [ ] Imports work: `python -c "from cortexgit import CortexGit"`
- [ ] README is clear and accurate
- [ ] setup.py and pyproject.toml are in sync
- [ ] LICENSE file exists (MIT recommended)
- [ ] .gitignore blocks build artifacts
- [ ] GitHub Actions workflows pass
- [ ] You've tested installation in a clean venv

---

## Commands You'll Run Repeatedly

```bash
# Development
pip install -e .[dev]
pytest tests/ -v
black src/ tests/
ruff check src/ tests/

# Release (one time, then repeat with new version)
python -m build
twine upload dist/*

# Update version before next release
# 1. Edit setup.py, pyproject.toml, __init__.py
# 2. git tag v0.X.0
# 3. python -m build && twine upload dist/*
```

---

## What Comes After Library Release

Once it's on PyPI and people start using it:

1. **Set up discussions** — GitHub Discussions for questions
2. **Monitor issues** — Fix bugs quickly, close duplicates
3. **Gather feedback** — Use real-world usage to inform v0.2.0
4. **Plan roadmap** — Document what's coming next
5. **Consider monetization** — You could offer:
   - Hosted version (pay for cloud compute)
   - Premium features (multi-tenant, analytics)
   - Support contract

But first: get v0.1.0 out, get users, learn from them.

---

## One Last Thing

Your README should have a badge that shows the installation command:

```markdown
## Installation

```bash
pip install cortexgit
```

Or in your GitHub releases page:
```bash
pip install cortexgit==0.1.0
```

The moment you publish, this command works for anyone worldwide. That's powerful.
```

You're done when people can do `pip install cortexgit` and it works.
