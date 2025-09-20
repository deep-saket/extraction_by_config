# Coding Style Guidelines

These guidelines are recommended for all future projects in this codebase. They are based on best practices and tailored to your current architecture.

## 1. File & Directory Structure
- Use `snake_case` for file and directory names.
- Group related modules into packages (e.g., `common/`, `models/`, `extraction_io/`).
- Place tests in a dedicated `tests/` directory, mirroring the source structure.

## 2. Naming Conventions
- Use `PascalCase` for class names (e.g., `BaseComponent`).
- Use `snake_case` for functions, variables, and file names.
- Use `UPPER_CASE` for constants and environment variables.

## 3. Imports
- Standard library imports first, then third-party, then local imports.
- Use absolute imports within the project.

## 4. Type Hints & Annotations
- Use type hints for all function arguments and return types.
- Use `Optional`, `List`, `Dict`, etc., from `typing` as needed.

## 5. Docstrings & Comments
- Add module-level docstrings for every file.
- Use Google or NumPy style docstrings for functions and classes.
- Comment on complex logic, but avoid redundant comments.

## 6. Logging
- Use the `logging` module for all logs.
- Prefer class-level loggers named after the class/module.
- Avoid using `print()` for logging.

## 7. Error Handling
- Use specific exception types where possible.
- Add informative error messages and consider custom exceptions for common error scenarios.

## 8. Configuration & Environment
- Store configuration in YAML or JSON files, loaded at runtime.
- Use environment variables for secrets and paths; provide a `.env.example` file.

## 9. Testing
- Write unit tests for all core logic.
- Use mocks for external dependencies (e.g., file I/O, model loading).
- Use `pytest` as the test runner.

## 10. Formatting & Linting
- Use `black` for code formatting.
- Use `flake8` or `ruff` for linting.
- Run formatters and linters in CI/CD if possible.

## 11. Dependency Management
- List all dependencies in `requirements.txt`.
- Use a lock file (e.g., `requirements.lock` or `poetry.lock`) for reproducibility.

## 12. API Design
- Use FastAPI for REST APIs.
- Validate all inputs using Pydantic models.
- Return informative error responses with status codes.

## 13. Documentation
- Maintain a `README.md` with setup, usage, and architecture overview.
- Add usage examples and diagrams where helpful.

---

_Adhering to these guidelines will help ensure code quality, maintainability, and ease of collaboration._

