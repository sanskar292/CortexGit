# setup.py
from setuptools import setup, find_packages

setup(
    name="cortexgit",
    version="0.1.0",
    description="Persistent memory for LLM agents. Event sourcing + semantic retrieval.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Antigravity",
    author_email="antigravity@google.com",
    url="https://github.com/google-deepmind/cortexgit",
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
        "Bug Reports": "https://github.com/google-deepmind/cortexgit/issues",
        "Source": "https://github.com/google-deepmind/cortexgit",
        "Documentation": "https://cortexgit.readthedocs.io",
    },
)
