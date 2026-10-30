# Opsy - Your AI-Powered SRE Colleague

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/datolabs-io/opsy/.github%2Fworkflows%2Fci.yaml?label=CI)
 ![GitHub Release](https://img.shields.io/github/v/release/datolabs-io/opsy?label=Release)

Opsy is an intelligent command-line assistant designed for Site Reliability Engineers (SREs), DevOps professionals, and platform engineers. It uses AI to help you navigate operational challenges, troubleshoot issues, and automate routine workflows. Opsy integrates with your existing tools and provides contextual assistance to make your daily operations more efficient.

Opsy uses a "tools-as-agents" architecture where each tool functions as a specialized AI agent with expertise in its domain (Kubernetes, Git, AWS, etc.). The main Opsy agent orchestrates these specialized agents, breaking down complex tasks and delegating them to the appropriate tools. This approach provides domain-specific expertise, improved safety through tool-specific validation, better context management for multi-step operations, and modular extensibility for adding new capabilities.

> [!WARNING]
> Opsy is currently in early development. While the core functionality works well, some features are still being refined. We recommend using it in non-production environments for now. We welcome your feedback to help improve Opsy.

## Demo

[![Opsy Demo](https://github.com/user-attachments/assets/19e27126-baa9-432e-a014-50a18d165fde)](https://youtu.be/j5sWZDvTFtA)

The demo above shows Opsy handling this complex task:

> Analyze the pods in the current namespace. If there are any pods that are failing, I need you to analyze the reason it is failing. Then, create a single Jira task named `Kubernetes issues` in `OPSY` project reporting the issue. The task description must contain your analysis for on the failing pods. In addition, I want to have backups for our deployments: extract the deployment manifests and push them into a new private `backup` repo in `datolabs-io-sandbox`.

Click on the screenshot to [watch the full demonstration](https://youtu.be/j5sWZDvTFtA).

## Prerequisites

### Anthropic API Key

Opsy uses Anthropic's Claude AI models to provide intelligent assistance. You'll need an Anthropic API key:

1. Create an account at [Anthropic's website](https://www.anthropic.com/)
2. Generate an API key from your account dashboard
3. Set the API key in your Opsy configuration (see Configuration section) or as an environment variable:

   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

### Command-Line Tools

Opsy works with standard [command-line tools](./assets/tools/). While none are strictly required to run Opsy, having them installed expands its capabilities:

- [Git](https://git-scm.com/downloads) - Version control
- [GitHub CLI](https://cli.github.com) - GitHub integration
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes management
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) - AWS management
- [Helm](https://helm.sh/docs/intro/install/) - Kubernetes package manager
- [Google Cloud CLI (gcloud)](https://cloud.google.com/sdk/docs/install) - Google Cloud management
- [Jira CLI](https://github.com/ankitpokhrel/jira-cli) - Jira automation

Opsy adapts to your environment and only uses tools that are installed on your system.

## Installation

### Via Go Install

For users with Go 1.24 or later:

```bash
go install github.com/datolabs-io/opsy/cmd/opsy@latest
```

Ensure your Go bin directory is in your PATH.

### Via Homebrew

For macOS and Linux users with [Homebrew](https://brew.sh):

```bash
brew install --cask datolabs-io/tap/opsy
```

### Direct Download

Each [release](https://github.com/datolabs-io/opsy/releases) includes binaries for various platforms:

1. Download the appropriate binary for your operating system
2. Make it executable (Unix-based systems): `chmod +x opsy`
3. Move it to a directory in your `PATH`: `mv opsy /usr/local/bin/` (or another directory in your `PATH`)

## Usage

Opsy is simple to use. Just describe what you want to do in plain language, and Opsy will handle the rest.

```bash
opsy 'Your task description here'
```

For example:

```bash
# Repository management
opsy 'Create a new private repository in datolabs-io organization named backup'

# Kubernetes troubleshooting
opsy 'Check why pods in the production namespace are crashing'

# Log analysis
opsy 'Find errors in the application logs from the last hour'
```

Opsy interprets your instructions, builds a plan, and executes the necessary actions to complete your task—no additional input required.

## Configuration

Opsy is configured via a YAML file located at `~/.opsy/config.yaml`:

```yaml
# UI configuration
ui:
  # Theme for the UI (default: "default")
  theme: default

# Logging configuration
logging:
  # Path to the log file (default: "~/.opsy/log.log")
  path: ~/.opsy/log.log
  # Logging level: debug, info, warn, error (default: "info")
  level: info

# Anthropic API configuration
anthropic:
  # Your Anthropic API key (required)
  api_key: your_api_key_here
  # Model to use (default: "claude-3-7-sonnet-latest")
  model: claude-3-7-sonnet-latest
  # Temperature for generation (default: 0.5)
  temperature: 0.5
  # Maximum tokens to generate (default: 1024)
  max_tokens: 1024

# Tools configuration
tools:
  # Maximum duration in seconds for a tool to execute (default: 120)
  timeout: 120
  # Exec tool configuration
  exec:
    # Timeout for exec tool (0 means use global timeout) (default: 0)
    timeout: 0
    # Shell to use for execution (default: "/bin/bash")
    shell: /bin/bash
```

You can also set configuration using environment variables with the prefix `OPSY_` followed by the configuration path in uppercase with underscores:

```bash
# Set the logging level
export OPSY_LOGGING_LEVEL=debug

# Set the tools timeout
export OPSY_TOOLS_TIMEOUT=180
```

The Anthropic API key can also be set via `ANTHROPIC_API_KEY` (without the `OPSY_` prefix).

## Extending & Contributing

We welcome contributions to Opsy! The project is designed to be easily extended.

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please update tests as appropriate and follow the existing coding style.

Here's how you can extend Opsy's capabilities:

### System Prompts

System prompts in [./assets/prompts](./assets/prompts/) define how Opsy understands and responds to user tasks:

#### Agent System Prompt

The primary prompt ([assets/prompts/agent_system.tmpl](./assets/prompts/agent_system.tmpl)) guides Opsy's overall behavior, establishing its identity as an AI assistant for SREs and DevOps professionals and defining the format for execution plans.

#### Tool System Prompt

This prompt ([assets/prompts/tool_system.tmpl](./assets/prompts/tool_system.tmpl)) defines how Opsy interacts with external tools, ensuring interactions are safe, effective, and follow best practices.

#### Tool User Prompt

This prompt ([assets/prompts/tool_user.tmpl](./assets/prompts/tool_user.tmpl)) defines the format for requesting tool execution, maintaining consistency in how tools are invoked.

To contribute a new prompt or modify an existing one, add it to the repository and submit a pull request.

### Tools

Tool definitions in [assets/tools/](./assets/tools/) allow Opsy to interact with various systems and services:

```yaml
---
display_name: Tool Name
executable: command-name
description: Description of what the tool does
inputs:
  parameter1:
    type: string
    description: Description of the first parameter
    default: "default-value"  # Optional default value
    examples:
      - "example1"
      - "example2"
    optional: false  # Whether this parameter is required
rules:
  - 'Rule 1 for using this tool'
  - 'Rule 2 for using this tool'
```

### Themes

Theme definitions in [assets/themes/](./assets/themes/) control Opsy's visual appearance:

```yaml
base:
  base00: "#1A1B26"  # Primary background
  base01: "#24283B"  # Secondary background
  base02: "#292E42"  # Borders and dividers
  base03: "#565F89"  # Muted text
  base04: "#A9B1D6"  # Primary text

accent:
  accent0: "#FF9E64"  # Command text
  accent1: "#9ECE6A"  # Agent messages
  accent2: "#7AA2F7"  # Tool output
```

## Acknowledgments

- [Charm](https://github.com/charmbracelet) for their TUI libraries
- [Anthropic](https://github.com/anthropics/anthropic-sdk-go) for their Go SDK for Claude AI models
- [Viper](https://github.com/spf13/viper) for configuration management
- Various Go libraries for schema validation, data structures, and YAML parsing
- The Go community for excellent tooling and support

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
