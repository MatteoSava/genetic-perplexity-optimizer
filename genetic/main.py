import pandas as pd
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM

from genetic.utils import set_seed, DEVICE, MODEL_NAME, SUBMISSION_FILE, OUTPUT_FILE
from genetic.enhanced_genome import EnhancedGeneticOptimizer

# Set random seed
set_seed()

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def generate_submission(hyperparams):
    """
    Reorders text in sample_submission.csv using the enhanced genetic algorithm
    and saves the result to submission.csv. Logs perplexity to wandb.
    """
    # Load the sample submission
    sample_submission = pd.read_csv(SUBMISSION_FILE)
    submission = sample_submission.copy()
    total_perplexity = 0

    # Process each row of the dataset
    for i, row in sample_submission.iterrows():
        # Skip some lines
        if i < 1:
            continue

        original_text = row["text"]
        words = original_text.split()

        # Handle empty or trivial cases
        if not words:
            submission.loc[i, "text"] = ""
            continue

        # Initialize optimizer with hyperparameters from wandb
        optimizer = EnhancedGeneticOptimizer(
            model,
            tokenizer,
            population_size=hyperparams["population_size"],
            elite_ratio=hyperparams["elite_ratio"],
            max_age=hyperparams["max_age"],
            mutation_rate=hyperparams["mutation_rate"],
            tournament_size=hyperparams["tournament_size"],
            device=DEVICE,
            batch_size=hyperparams["batch_size"],
            row_id=int(i),
        )

        # Optimize the word order using the enhanced GA
        optimized_text, perplexity = optimizer.optimize(
            words, generations=hyperparams["generations"]
        )

        # Update submission with reordered text
        submission.loc[i, "text"] = optimized_text
        total_perplexity += perplexity

        # Log perplexity per row
        wandb.log({f"Perplexity {i}": perplexity})
        wandb.log({f"Optimized Text {i}": optimized_text})

        print(f"Row {i + 1}/{len(sample_submission)}:")
        print(f"  Optimized Text: {optimized_text}")
        print(f"  Perplexity: {perplexity}")

    # Compute average perplexity
    average_perplexity = total_perplexity / len(sample_submission)

    # Log average perplexity
    wandb.log({"average_perplexity": average_perplexity})

    # Save the submission file
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")


def train():
    # Define hyperparameters
    hyperparams = {
        "population_size": 150,
        "elite_ratio": 0.1,
        "max_age": 15,
        "generations": 125,
        "mutation_rate": 0.6,
        "tournament_size": 8,
        "batch_size": 256,
    }

    # Initialize wandb
    wandb.init(project="santa-2024-kaggle", config=hyperparams)

    generate_submission(hyperparams)


# Main Execution
if __name__ == "__main__":
    train()
