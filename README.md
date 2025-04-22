# Original Scheduler

## Overview

The Original Scheduler project is a distributed machine learning framework designed to facilitate federated learning using various neural network models. It leverages PyTorch for model training and multiprocessing for parallel execution across multiple devices.

## Features

- **Federated Learning**: Supports federated learning with customizable client-server architecture.
- **Neural Network Models**: Includes implementations of various models such as ResNet, MobileNet, SqueezeNet, VGG, and GPT-2.
- **Data Loading**: Provides flexible data loading mechanisms with support for custom datasets.
- **Multiprocessing**: Utilizes Python's multiprocessing to handle multiple training and testing processes concurrently.
- **Model Checkpointing**: Automatically saves model checkpoints during training for later evaluation.

## Project Structure

- `src/`: Contains the main source code for the project.
  - `main.py`: The main entry point for running the federated learning experiments.
  - `arguments.py`: Handles command-line arguments for configuring experiments.
  - `node.py`: Defines the client node behavior in the federated learning setup.
  - `server.py`: Implements the server logic for aggregating model updates.
  - `workers.py`: Contains worker functions for training and testing models.
  - `dataLoader/`: Includes data loading utilities and dataset definitions.
  - `nn_models/`: Houses various neural network model implementations.
    - `transformers/`: Contains transformer models like GPT-2.
  - `sh_scripts/`: Shell scripts for automating tasks.
  - `baseline/`: Baseline models and configurations.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd original_scheduler
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To start a federated learning experiment, run the following command:

```bash
python src/main.py --nodes <num_nodes> --round <num_rounds> --dataset <dataset_name>
```

### Configuration

Experiment configurations can be adjusted via command-line arguments or by modifying the `arguments.py` file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- This project uses the GPT-2 model from the Hugging Face Transformers library.
- Special thanks to the contributors and the open-source community.

