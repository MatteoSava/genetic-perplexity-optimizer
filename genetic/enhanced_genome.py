import collections
import random
import torch
import numpy as np
from tqdm.auto import tqdm
import wandb
from genetic.utils import DEVICE

from dataclasses import dataclass
from typing import List


@dataclass
class EnhancedGenome:
    sequence: List[str]
    fitness: float = float("-inf")
    age: int = 0

    def text(self) -> str:
        return " ".join(self.sequence)


class EnhancedGeneticOptimizer:
    def __init__(
        self,
        model,
        tokenizer,
        population_size=100,
        elite_ratio=0.1,
        max_age=20,
        mutation_rate=0.3,
        tournament_size=3,
        batch_size=16,
        device=DEVICE,
        row_id=None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_ratio))
        self.max_age = max_age
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.batch_size = batch_size
        self.device = device
        self.row_id = row_id

    def calculate_genome_diversity(self, population):
        from itertools import combinations

        total_similarity = 0
        for genome1, genome2 in combinations(population, 2):
            set1, set2 = set(genome1.sequence), set(genome2.sequence)
            total_similarity += len(set1.intersection(set2)) / float(
                len(set1.union(set2))
            )
        avg_similarity = total_similarity / (
            len(population) * (len(population) - 1) / 2
        )
        diversity = 1 - avg_similarity  # Diversity = 1 - similarity
        wandb.log({f"Diversity {self.row_id}": diversity})
        return diversity

    def calculate_entropy(self, population):
        fitness_values = [genome.fitness for genome in population]
        total_count = len(fitness_values)
        from collections import Counter

        fitness_counts = Counter(fitness_values)
        probabilities = np.array(
            [count / total_count for count in fitness_counts.values()]
        )
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Avoid log(0)
        wandb.log({f"Entropy {self.row_id}": entropy})
        return entropy

    def adaptive_mutation_rate(
        self, entropy, diversity, base_rate=0.3, min_rate=0.1, max_rate=0.7
    ):
        if entropy < 0.5 or diversity < 0.3:  # Low randomness or diversity
            return min(max_rate, base_rate + 0.2)
        return max(min_rate, base_rate - 0.1)

    def reinitialize_population(
        self,
        population,
        diversity,
        entropy,
        threshold_diversity=0.2,
        threshold_entropy=0.3,
    ):
        if diversity < threshold_diversity and entropy < threshold_entropy:
            num_to_reinitialize = int(
                self.population_size * 0.1
            )  # Reinitialize 10% of the population
            for _ in range(num_to_reinitialize):
                index = random.randint(0, len(population) - 1)
                population[index] = EnhancedGenome(
                    sequence=random.sample(
                        population[index].sequence, len(population[index].sequence)
                    )
                )
                population[index].age = 0
            wandb.log({"Reinitialized Individuals": num_to_reinitialize})
        return population

    def check_reinitialization(self, cumulative_improvement, diversity, threshold=0.1):
        if cumulative_improvement < threshold and diversity < 0.2:
            return True
        return False

    def calculate_fitness_variance(self, population):
        """
        Calculate the variance of fitness values in the population.
        """
        fitness_values = [genome.fitness for genome in population]
        variance = np.var(fitness_values)  # NumPy calculates variance
        wandb.log({f"Fitness Variance {self.row_id}": variance})  # Log to WandB
        return variance

    def adjust_mutation_rate(
        self, population, base_rate=0.05, min_rate=0.1, max_rate=0.8
    ):
        diversity = self.calculate_genome_diversity(population)
        new_mutation_rate = min_rate + (max_rate - min_rate) * diversity
        wandb.log({f"Mutation Rate {self.row_id}": new_mutation_rate})
        return new_mutation_rate

    def _evaluate_fitness(self, genomes: List[EnhancedGenome]):
        """
        Evaluate the fitness of each genome based on perplexity (negative log-likelihood)
        using batched processing for improved performance.
        """
        for i in range(0, len(genomes), self.batch_size):
            batch_genomes = genomes[i : i + self.batch_size]

            # Prepare batch texts with special tokens
            batch_texts = [
                f"{self.tokenizer.bos_token}{genome.text()}{self.tokenizer.eos_token}"
                for genome in batch_genomes
            ]

            # Tokenize the batch without adding special tokens (already added)
            batch_inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,  # Disable truncation
                add_special_tokens=False,
            ).to(self.device)

            # Remove 'token_type_ids' if present
            if "token_type_ids" in batch_inputs:
                batch_inputs.pop("token_type_ids")

            # Create attention mask for padded sequences
            attention_mask = batch_inputs["attention_mask"]

            with torch.no_grad():
                outputs = self.model(**batch_inputs, use_cache=False)

            # Shift logits and labels for causal language modeling
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch_inputs["input_ids"][..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Reshape loss to match sequence lengths
            loss = loss.view(shift_labels.size())

            # Apply attention mask to ignore padding tokens
            masked_loss = loss * shift_attention_mask

            # Calculate average loss per sequence
            sequence_lengths = shift_attention_mask.sum(dim=1)
            sequence_losses = masked_loss.sum(dim=1) / sequence_lengths

            # Assign negative losses as fitness scores
            for genome, loss_val in zip(batch_genomes, sequence_losses):
                genome.fitness = -loss_val.item()

    def _initialize_population(self, words: List[str]) -> List[EnhancedGenome]:
        """
        Initialize the population with random permutations of the words.
        """
        population = [
            EnhancedGenome(sequence=random.sample(words, len(words)))
            for _ in range(self.population_size)
        ]
        return population

    def _tournament_select(self, population: List[EnhancedGenome]) -> EnhancedGenome:
        """
        Select an individual from the population using tournament selection.
        """
        return max(
            random.sample(population, self.tournament_size), key=lambda x: x.fitness
        )

    def _adaptive_crossover(
        self,
        parent1: EnhancedGenome,
        parent2: EnhancedGenome,
        generation: int,
        total_generations: int,
    ) -> EnhancedGenome:
        """
        Perform adaptive crossover between two parents to produce a child genome, handling duplicates correctly.
        """
        # Adaptive crossover rate decreases over generations
        crossover_rate = 0.7 * (1 - generation / total_generations) + 0.3 * (
            generation / total_generations
        )
        if random.random() < crossover_rate:
            length = len(parent1.sequence)
            # Choose crossover points
            if length > 1:
                cut1 = random.randint(0, length - 2)
                cut2 = random.randint(cut1 + 1, length)
            else:
                cut1, cut2 = 0, length
            # Copy a slice from parent1
            child_sequence = [None] * length
            child_sequence[cut1:cut2] = parent1.sequence[cut1:cut2]
            # Now fill in the rest from parent2, maintaining word counts
            child_counter = collections.Counter(parent1.sequence[cut1:cut2])
            parent1_counter = collections.Counter(parent1.sequence)
            p2_index = 0
            for i in range(length):
                if child_sequence[i] is None:
                    while p2_index < length:
                        word = parent2.sequence[p2_index]
                        p2_index += 1
                        # Check if we can add this word without exceeding counts
                        if child_counter[word] < parent1_counter[word]:
                            child_sequence[i] = word
                            child_counter[word] += 1
                            break
                    else:
                        # If no more words from parent2, fill from parent1
                        for word in parent1.sequence:
                            if child_counter[word] < parent1_counter[word]:
                                child_sequence[i] = word
                                child_counter[word] += 1
                                break
        else:
            # Simple crossover: shuffle one of the parents
            child_sequence = parent1.sequence.copy()
            random.shuffle(child_sequence)
        # Ensure valid permutation
        if collections.Counter(child_sequence) != collections.Counter(parent1.sequence):
            child_sequence = parent1.sequence.copy()
            random.shuffle(child_sequence)
        return EnhancedGenome(sequence=child_sequence)

    def _mutate(self, genome: EnhancedGenome) -> EnhancedGenome:
        """
        Mutate the genome by swapping two words or reinitializing if it has aged.
        """
        if genome.age > self.max_age:
            # Reinitialize genome if it's too old
            random.shuffle(genome.sequence)
            genome.age = 0
        else:
            # Mutation occurs based on mutation_rate
            if random.random() < self.mutation_rate:
                if len(genome.sequence) > 1:
                    i, j = random.sample(range(len(genome.sequence)), 2)
                    genome.sequence[i], genome.sequence[j] = (
                        genome.sequence[j],
                        genome.sequence[i],
                    )
        return genome

    def optimize(self, words: List[str], generations: int = 100):
        population = self._initialize_population(words)

        # Initialize fitness
        self._evaluate_fitness(population)
        best_fitness = max(population, key=lambda x: x.fitness).fitness

        # Create progress bar for generations
        pbar = tqdm(range(1, generations + 1), desc="Generations", position=0)

        for gen in pbar:
            # Increase age for all genomes
            for genome in population:
                genome.age += 1

            # Evaluate fitness
            self._evaluate_fitness(population)
            # Calculate metrics
            diversity = self.calculate_genome_diversity(population)
            self.calculate_fitness_variance(population)
            entropy = self.calculate_entropy(population)
            # Reinitialize part of the population if necessary
            population = self.reinitialize_population(population, diversity, entropy)
            # Update mutation rate
            self.mutation_rate = self.adaptive_mutation_rate(entropy, diversity)
            # Sort population by fitness
            population = sorted(population, key=lambda x: x.fitness, reverse=True)

            # Elitism: retain top elites
            elites = population[: self.elite_size]

            # Generate new population
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                # Using current generation number
                current_gen = gen
                child = self._adaptive_crossover(
                    parent1, parent2, current_gen, generations
                )
                # Mutation
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

            # Update best fitness
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness

            # Update progress bar with best fitness
            pbar.set_postfix({"Best Fitness": f"{-best_fitness:.4f}"})

            # Log metrics
            wandb.log(
                {
                    f"Generation {self.row_id}": gen,
                    f"Best Fitness {self.row_id}": -best_fitness,
                    f"Mutation Rate {self.row_id}": self.mutation_rate,
                }
            )

        # Final evaluation to get the best genome
        self._evaluate_fitness(population)
        best_genome = max(population, key=lambda x: x.fitness)

        # Compute perplexity
        perplexity = np.exp(-best_genome.fitness)

        return best_genome.text(), perplexity
