# Contributing

Thank you for your interest in contributing to the COCO to 3D Synthetic Pipeline!

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/coco-to-3d-synthetic.git
cd coco-to-3d-synthetic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

This project follows PEP 8 style guidelines.

### Formatting

Use Black for code formatting:

```bash
black src/
```

### Linting

Run Flake8 to check for issues:

```bash
flake8 src/
```

### Type Checking

Use mypy for type checking:

```bash
mypy src/
```

## Testing

Write tests for new features and bug fixes.

### Running Tests

```bash
pytest tests/
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_coco_processing.py
from src.coco_processing import COCOProcessor

def test_load_annotations():
    processor = COCOProcessor('test_data/annotations.json')
    annotations = processor.load_annotations()
    assert annotations is not None
    assert 'images' in annotations
```

## Pull Request Process

1. **Create a branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/add-new-renderer
   ```

2. **Make changes**: Follow code style guidelines

3. **Add tests**: Ensure your changes are tested

4. **Update documentation**: Update relevant docs in `docs/`

5. **Commit**: Write clear commit messages
   ```bash
   git commit -m "Add support for GLB model export"
   ```

6. **Push**: Push to your fork
   ```bash
   git push origin feature/add-new-renderer
   ```

7. **Submit PR**: Open a pull request on GitHub

## Commit Message Guidelines

Use clear, descriptive commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

Examples:

```
Add support for GLB format export

Implement GLB export functionality in ModelGenerator.
Closes #123
```

## Documentation

Update documentation when:

- Adding new features
- Changing APIs
- Fixing bugs that affect usage

Documentation is in the `docs/` directory and built with MkDocs.

### Building Docs Locally

```bash
mkdocs serve
```

Visit http://127.0.0.1:8000 to preview.

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

### Reporting Issues

Report issues or bugs by opening a GitHub issue with:

- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

## Feature Requests

We welcome feature requests! Open an issue with:

- Clear description of the feature
- Use cases and motivation
- Possible implementation approach (optional)

## Questions?

Feel free to open an issue for questions or discussion.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
