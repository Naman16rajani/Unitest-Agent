"""
Configuration module for LLM-Optimized CrewAI Unit Test Generation System
"""

import os
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # If dotenv is not available, continue without it
    pass


class LLMProvider(Enum):
    """Available LLM providers"""

    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"
    AZURE = "azure"


class TestingFramework(Enum):
    """Supported testing frameworks"""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    JUNIT = "junit"
    MOCHA = "mocha"
    VITEST = "vitest"


class DocumentationLevel(Enum):
    """Documentation detail levels"""

    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 30


@dataclass
class TestingConfig:
    """Configuration for testing parameters"""

    framework: TestingFramework = TestingFramework.PYTEST
    coverage_target: float = 80.0
    max_retry_attempts: int = 3
    parallel_execution: bool = True
    performance_testing: bool = False
    security_testing: bool = False
    documentation_level: DocumentationLevel = DocumentationLevel.DETAILED
    test_types: list = field(default_factory=lambda: ["unit"])
    include_integration_tests: bool = False
    include_edge_cases: bool = True
    mock_external_dependencies: bool = True


@dataclass
class SystemConfig:
    """Main system configuration"""

    llm: LLMConfig
    testing: TestingConfig
    output_directory: str = "./generated_tests"
    source_directory: str = "./sample_code"
    verbose: bool = True
    debug: bool = False


# Predefined LLM configurations
LLM_CONFIGS = {
    LLMProvider.GEMINI: {
        "model": os.getenv("GOOGLE_MODEL_ID", "gemini-1.5-flash"),
        "api_key_env": "GOOGLE_API_KEY",
        "temperature": float(os.getenv("GOOGLE_MODEL_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("GOOGLE_MODEL_MAX_TOKENS", "4096")),
    },
    LLMProvider.OPENAI: {
        "model": "gpt-4-turbo-preview",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    LLMProvider.CLAUDE: {
        "model": "claude-3-sonnet-20240229",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    LLMProvider.OLLAMA: {
        "model": "codellama:13b",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
}

# Framework-specific configurations
FRAMEWORK_CONFIGS = {
    TestingFramework.PYTEST: {
        "file_pattern": "test_*.py",
        "imports": ["import pytest", "from unittest.mock import Mock, patch"],
        "decorators": ["@pytest.fixture", "@pytest.mark.parametrize"],
        "assertions": "assert",
    },
    TestingFramework.UNITTEST: {
        "file_pattern": "test_*.py",
        "imports": ["import unittest", "from unittest.mock import Mock, patch"],
        "decorators": ["@unittest.mock.patch"],
        "assertions": "self.assertEqual",
    },
    TestingFramework.JEST: {
        "file_pattern": "*.test.js",
        "imports": ["const { jest } = require('@jest/globals');"],
        "decorators": ["describe", "it", "beforeEach"],
        "assertions": "expect",
    },
}


def get_default_config(llm_provider: LLMProvider = LLMProvider.GEMINI, output_directory: str = "./generated_tests") -> SystemConfig:
    """Get default system configuration with specified LLM provider"""
    llm_config_data = LLM_CONFIGS[llm_provider]

    llm_config = LLMConfig(
        provider=llm_provider,
        model=llm_config_data["model"],
        api_key=os.getenv(llm_config_data.get("api_key_env")),
        base_url=llm_config_data.get("base_url"),
        temperature=llm_config_data["temperature"],
        max_tokens=llm_config_data["max_tokens"],
    )

    testing_config = TestingConfig()

    return SystemConfig(llm=llm_config, testing=testing_config, output_directory=output_directory)


def update_llm_provider(config: SystemConfig, provider: LLMProvider) -> SystemConfig:
    """Update the LLM provider in an existing configuration"""
    llm_config_data = LLM_CONFIGS[provider]

    config.llm.provider = provider
    config.llm.model = llm_config_data["model"]
    config.llm.api_key = os.getenv(llm_config_data.get("api_key_env"))
    config.llm.base_url = llm_config_data.get("base_url")
    config.llm.temperature = llm_config_data["temperature"]
    config.llm.max_tokens = llm_config_data["max_tokens"]

    return config


def validate_config(config: SystemConfig) -> bool:
    """Validate configuration settings"""
    if not config.llm.api_key and config.llm.provider != LLMProvider.OLLAMA:
        print(f"Warning: No API key found for {config.llm.provider.value}")
        return False

    if config.testing.coverage_target < 0 or config.testing.coverage_target > 100:
        print("Error: Coverage target must be between 0 and 100")
        return False

    if not os.path.exists(config.source_directory):
        print(f"Error: Source directory {config.source_directory} does not exist")
        return False

    return True


def create_llm_instance(llm_config: LLMConfig) -> Any:
    """Create and return a LangChain LLM instance based on configuration"""

    if llm_config.provider == LLMProvider.GEMINI:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # For Google, try different environment variable names if API key is None
            api_key = llm_config.api_key
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")

            if not api_key:
                raise ValueError(
                    "No Google API key found. Please set GOOGLE_API_KEY environment variable. "
                    "Get your API key from https://aistudio.google.com/app/apikey"
                )

            # Map common model names to current ones
            model_name = llm_config.model
            # if model_name == "gemini-pro":
            #     # model_name = "gemini-1.5-flash"
            #     print("⚠️  gemini-pro is deprecated, using gemini-1.5-flash instead")
            # elif model_name == "gemma-3-27b-it":
            #     # This model might not be available via API, try gemini instead
            #     # model_name = "gemini-1.5-flash"
            #     print(
            #         "⚠️  gemma-3-27b-it may not be available via API, using gemini-1.5-flash instead"
            #     )

            print(f"🤖 Using model: {model_name}")
            print(f"🔑 API key configured: {'Yes' if api_key else 'No'}")

            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=llm_config.temperature,
                max_output_tokens=llm_config.max_tokens,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain-google-genai: pip install langchain-google-genai"
            )
        except Exception as e:
            available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
            raise ValueError(
                f"Error creating Gemini model: {str(e)}\n"
                f"Available models: {', '.join(available_models)}\n"
                f"Your model: {llm_config.model}\n"
                f"Try updating your GOOGLE_MODEL_ID in .env file to one of the available models."
            )

    # elif llm_config.provider == LLMProvider.OPENAI:
    #     try:
    #         from langchain_openai import ChatOpenAI

    #         return ChatOpenAI(
    #             model=llm_config.model or "gpt-3.5-turbo",
    #             api_key=llm_config.api_key,
    #             temperature=llm_config.temperature,
    #             max_tokens=llm_config.max_tokens,
    #         )
    #     except ImportError:
    #         raise ImportError(
    #             "Please install langchain-openai: pip install langchain-openai"
    #         )

    # elif llm_config.provider == LLMProvider.CLAUDE:
    #     try:
    #         from langchain_anthropic import ChatAnthropic

    #         return ChatAnthropic(
    #             model=llm_config.model or "claude-3-sonnet-20240229",
    #             api_key=llm_config.api_key,
    #             temperature=llm_config.temperature,
    #             max_tokens=llm_config.max_tokens,
    #         )
    #     except ImportError:
    #         raise ImportError(
    #             "Please install langchain-anthropic: pip install langchain-anthropic"
    #         )

    # elif llm_config.provider == LLMProvider.OLLAMA:
    #     try:
    #         from langchain_community.llms import Ollama

    #         return Ollama(
    #             model=llm_config.model or "codellama",
    #             base_url=llm_config.base_url or "http://localhost:11434",
    #             temperature=llm_config.temperature,
    #         )
    #     except ImportError:
    #         raise ImportError(
    #             "Please install langchain-community: pip install langchain-community"
    #         )

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
