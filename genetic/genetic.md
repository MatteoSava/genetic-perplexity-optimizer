**`utils.py`**

Contains utility functions and global configurations used throughout the project.


**`enhanced_genome.py`**

Implements the genetic algorithm logic for optimizing word order to minimize perplexity.

**Key Components**:
 - **`EnhancedGenome` Class**: A data class representing an individual in the population, containing the sequence (word order), fitness score, and age.
 - **`EnhancedGeneticOptimizer` Class**: Encapsulates all methods related to the genetic algorithm.


**`main.py`**

The main script that orchestrates the execution of the project.

**Key Responsibilities**
 - **Model and Tokenizer Loading**: Loads the pre-trained language model and tokenizer specified in `utils.py`.
 - **Data Processing**: Reads the input data from `sample_submission.csv` and prepares it for optimization.
 - **Hyperparameter Handling**: Initializes hyperparameters, either from the default settings or from Weights & Biases (wandb) configurations.
 - **Optimization Execution**: Creates an instance of `EnhancedGeneticOptimizer` and runs the optimization for each text sequence.
 - **Results Logging**: Logs metrics like perplexity to wandb for tracking and analysis.
 - **Output Generation**: Saves the optimized text sequences to `submission.csv`.


**`config.yaml`**

Configuration file for Weights & Biases hyperparameter sweeps.
