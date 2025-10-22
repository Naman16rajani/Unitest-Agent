# UnitCraft 🧪✨

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


Code to run 
uv run python -m pytest --cov=sample_code --cov-report=html --cov-config=sample_code/.coveragerc                             

**UnitCraft** is an intelligent Python unit test generation system powered by Large Language Models (LLMs). It automatically analyzes your Python code and generates comprehensive, syntactically correct unit tests using state-of-the-art AI models.

## 🚀 Features

- **🤖 Multi-LLM Support**: Works with Google Gemini, OpenAI, Claude, Ollama, and Azure OpenAI
- **📝 Intelligent Code Analysis**: Automatically analyzes Python files and extracts testable methods
- **✅ Syntax Validation**: Built-in syntax checking ensures generated tests are valid Python code
- **🔄 Auto-retry Logic**: Automatically retries test generation if syntax errors are detected
- **🎯 Method-level Testing**: Generates focused unit tests for individual methods and functions
- **🛠️ Framework Support**: Works with pytest and unittest frameworks
- **📦 Easy Setup**: Quick installation with automated setup script

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Supported LLM Providers](#-supported-llm-providers)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Automated Setup

The easiest way to set up UnitCraft is using the provided setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
1. Install `uv` package manager (if not already installed)
2. Create a virtual environment
3. Install all dependencies
4. Set up the project structure

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/Naman16rajani/Unitest-Agent.git
cd Unitest-Agent

# Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root with your API keys:

```env
# Required: At least one LLM provider API key
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here

# Optional: Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

## 🚀 Quick Start

Generate unit tests for a Python file:

```bash
# Use default LLM (Gemini)
uv run main.py sample_code/calculator.py

# Specify a different LLM provider
uv run main.py sample_code/calculator.py --llm openai

# Custom output directory
uv run main.py sample_code/calculator.py --output ./tests
```

## 📖 Usage

### Command Line Interface

```bash
uv run main.py [FILE] [OPTIONS]
```

**Arguments:**
- `FILE`: Python file to generate tests for (default: `sample_code/calculator.py`)

**Options:**
- `--llm {gemini,openai,claude,ollama}`: LLM provider to use (default: gemini)
- `--output DIR`: Output directory for generated tests (default: `./generated_tests`)
- `--help`: Show help message

### Examples

```bash
# Generate tests for calculator.py using Gemini
uv run main.py sample_code/calculator.py

# Use OpenAI GPT-4
uv run main.py sample_code/complex.py --llm openai

# Use Claude with custom output
uv run main.py sample_code/tensorflow.py --llm claude --output ./my_tests

# Use local Ollama model
uv run main.py my_script.py --llm ollama
```

### Programmatic Usage

```python
from config import LLMProvider, create_llm_instance
from workflow import workflow

# Create LLM instance
llm = create_llm_instance(LLMProvider.GEMINI)

# Generate tests
result = workflow(llm, "path/to/your/file.py")
print(result)  # "unittest is generated" or error message
```

## ⚙️ Configuration

### LLM Providers

UnitCraft supports multiple LLM providers. Configure them in `config.py`:

**Google Gemini** (Default)
```python
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.7
```

**OpenAI**
```python
OPENAI_MODEL = "gpt-4"
OPENAI_TEMPERATURE = 0.7
```

**Claude (Anthropic)**
```python
CLAUDE_MODEL = "claude-3-sonnet-20240229"
CLAUDE_TEMPERATURE = 0.7
```

**Ollama (Local)**
```python
OLLAMA_MODEL = "llama2"
OLLAMA_BASE_URL = "http://localhost:11434"
```

### Testing Framework

Choose between pytest and unittest:

```python
from config import TestingFramework

config = get_default_config()
config.testing_framework = TestingFramework.PYTEST  # or TestingFramework.UNITTEST
```

### Advanced Configuration

Modify `config.py` to customize:
- Maximum retry attempts (`MAX_AGENT_FLOW_COUNT`)
- Model parameters (temperature, max tokens)
- Output directories
- Code analysis settings

## 📁 Project Structure

```
Unitest-Agent/
├── main.py                     # CLI entry point
├── config.py                   # Configuration and LLM setup
├── workflow.py                 # Main workflow orchestration
├── setup.sh                    # Automated setup script
├── pyproject.toml             # Project dependencies
├── README.md                   # This file
│
├── agents/                     # AI Agent implementations
│   ├── agent.py               # Base agent class
│   ├── prompt.py              # Prompt management
│   └── unittest_agent/
│       ├── unittest_agent.py         # Unit test generation agent
│       └── unittest_agent_prompt.py  # Test generation prompts
│
├── helpers/                    # Utility functions
│   ├── check_syntax.py        # Python syntax validation
│   ├── clean_code.py          # Code cleaning utilities
│   ├── prompt_writter.py      # Prompt formatting
│   ├── read_file.py           # File reading utilities
│   ├── save_code.py           # Save generated tests
│   └── analysers/
│       ├── code_analyzer.py                # Code analysis base
│       └── method_formatter_analyzer.py    # Method extraction
│
├── sample_code/                # Example Python files
│   ├── calculator.py          # Simple calculator example
│   ├── complex.py             # Complex example
│   └── tensorflow.py          # ML framework example
│
└── generated_tests/            # Output directory (auto-created)
```

## 💡 Examples

### Sample Input (`sample_code/calculator.py`)

```python
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
```

### Generated Output

UnitCraft will analyze the code and generate comprehensive unit tests including:
- ✅ Normal case tests
- ✅ Edge case tests
- ✅ Error handling tests
- ✅ Boundary condition tests

The generated tests are saved to `generated_tests/unittest.py` (or your specified output directory).

## 🤝 Supported LLM Providers

| Provider | Status | Model Examples |
|----------|--------|----------------|
| **Google Gemini** | ✅ Supported | gemini-1.5-flash, gemini-1.5-pro |
| **OpenAI** | ✅ Supported | gpt-4, gpt-3.5-turbo |
| **Claude (Anthropic)** | ✅ Supported | claude-3-opus, claude-3-sonnet |
| **Ollama** | ✅ Supported | llama2, mistral, codellama |
| **Azure OpenAI** | ✅ Supported | Custom deployments |

## 🔄 How It Works

1. **Code Analysis**: The `MethodFormatterAnalyzer` parses your Python file and extracts methods/functions
2. **Test Generation**: The `UnitTestAgent` uses an LLM to generate unit tests for each method
3. **Syntax Validation**: Generated tests are validated for Python syntax
4. **Auto-Retry**: If syntax errors are detected, the agent retries (up to 3 attempts)
5. **Output**: Valid tests are saved to the specified output directory

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_calculator.py
```

### Code Style

```bash
# Format code
black .

# Lint code
flake8 .
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
# Reinstall dependencies
uv pip install -e .
```

**"Invalid API key" errors**
- Check your `.env` file contains the correct API key
- Ensure the key is properly formatted without extra spaces

**"Syntax error in generated code"**
- The system will auto-retry up to 3 times
- Try a different LLM provider
- Check your input code is valid Python

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by leading LLM providers
- Inspired by the need for automated, intelligent test generation

## 📧 Contact

**Naman Rajani**  
GitHub: [@Naman16rajani](https://github.com/Naman16rajani)  
Repository: [Unitest-Agent](https://github.com/Naman16rajani/Unitest-Agent)

## 🌟 Star History

If you find UnitCraft helpful, please consider giving it a star on GitHub! ⭐

---

**Made with ❤️ by Naman Rajani**
