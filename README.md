# Reasoning Modules

Enhanced logical reasoning in AI systems using Tree of Thoughts (ToT) approach with Chain of Thought (CoT) prompting.

### Key Features

- **Multi-path Reasoning**: Generates multiple reasoning paths for a single problem
- **Chain of Thought Prompting**: Uses explicit CoT prompting to improve reasoning capabilities
- **Configurable System**: Easily customize prompts, LLM parameters, and deployment settings
- **Naptha SDK Integration**: Seamless integration with the Naptha agent framework
- **Flexible Deployment**: Run as a standalone module or deploy as a Naptha agent

## 🧩 How It Works

The Reasoning Modules system works by:

1. The system takes a reasoning problem (e.g., a mathematical proof) as input
2. Generates a configurable number of reasoning attempts using Chain of Thought prompting
3. Returns all reasoning paths in a structured format
4. Language models (default: GPT-4o-mini) for reasoning

This approach is inspired by research on Tree of Thoughts (ToT) methods, which show improved performance on complex reasoning tasks compared to single-path approaches.

## 🗂️ Project Structure

```
reasoning_modules/
├── __init__.py
├── configs/
│   ├── deployment.json       # Deployment configuration
│   └── llm_configs.json      # LLM configuration
├── prompt.py                 # Prompt templates
├── run.py                    # Main implementation
├── schemas.py                # Input/output schemas
└── test.jsonl                # Test examples
```

## Installation

### Prerequisites

- Python 3.12 (For all check the .toml file)
- Poetry
- Naptha SDK
- Pydantic

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/marathan24/reasoning_modules.git
   cd reasoning_modules
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   # Create a .env file with your keys
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   echo "NODE_URL=your_node_url" >> .env
   echo "PRIVATE_KEY=your_private_key" >> .env
   ```

## 📋 Usage

### Starting the Node (on your local machine)

You need to clone the node repository and start the node: [text](https://github.com/NapthaAI/naptha-node)

```bash
bash launch.sh
```
The node will start on localhost:7001 by default. If it doesn't, then adopt the following commands:


```bash
./docker-ctl.sh down
```

```bash
./launch.sh
```

Wait for some time until the node is up and running. You can check the logs using:

```bash
./docker-ctl.sh logs
```

You can also run the module directly using Python (after providing env variables):

```bash
poetry run python reasoning_modules/run.py
```

### Deploying as a Naptha Agent

Register the module as a Naptha agent:

```bash
naptha agents reasoning_modules -c "description='Reasoning modules' parameters='{"func_name": "str", "problem": "str", "num_thoughts": "int"}' module_url='https://github.com/marathan24/reasoning_modules'"
```

Release a versioned tag:

```bash
git tag v0.1
git push origin v0.1
```

Update the agent with the version (whichever is latest in repo):

```bash
naptha agents reasoning_modules -u "module_version='v0.2'" 
```

### Running the Agent

Execute the agent with specific parameters:

```bash
naptha run agent:reasoning_modules -p "func_name='reason', problem='Prove that if n is a positive integer, then the sum of the first n positive odd integers equals n^2', num_thoughts=3"
```

## 📊 Input Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `func_name` | string | The function name to execute | Required |
| `problem` | string | The reasoning problem to solve | Required |
| `num_thoughts` | integer | Number of reasoning paths to generate | 3 |

## 📝 Prompt Templates

The system includes several prompt templates for different reasoning tasks:

- **Standard Prompt**: Basic math problem prompt
- **CoT Prompt**: Chain of Thought prompt with strategy generation
- **Vote Prompt**: For selecting among multiple choices
- **Compare Prompt**: For comparing solution correctness
- **Score Prompt**: For evaluating solution quality

## 🔄 How the Code Works

1. **Initialization**: The `ReasoningAgent` class initializes with deployment configurations
2. **Input Processing**: The input problem and parameters are validated using Pydantic models
3. **Thought Generation**: Multiple reasoning paths are generated using the CoT prompt and LLM
4. **Response Collection**: All reasoning paths are collected and returned
5. **Integration**: The system integrates with Naptha SDK for agent deployment and execution

## Acknowledgements

- Naptha node and Naptha SDK
- Research on Tree of Thoughts and Chain of Thought prompting