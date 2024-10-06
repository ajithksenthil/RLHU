# rlhu_training.py

import os

# Disable CUDA devices if any and enable MPS fallback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Ensure fallback is enabled
os.environ["TOKENIZERS_PARALLELISM"] = "false"    # Suppress tokenizers parallelism warnings

# Now import other libraries
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader
import sys
import argparse
import warnings
from copy import deepcopy

# Suppress specific warnings if desired
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import your PsychologicalProfiler
from user_profile_project.psych_profiler_1_copy import PsychologicalProfiler  # Adjust the import path as needed

# Initialize the profiler
profiler = PsychologicalProfiler()

# Define the compute_certainty function
def compute_certainty(response: str, context: str = "general") -> float:
    _, certainty = profiler.update_profile(response, context)
    certainty = max(0.0, min(certainty, 0.99))
    return certainty

def parse_args():
    parser = argparse.ArgumentParser(description="RLHU Training Script")
    parser.add_argument('--device', type=str, default='mps', help="Device to use: 'cuda', 'cpu', 'mps'")
    return parser.parse_args()

def main():
    args = parse_args()

    # Print TRL version for debugging
    import trl
    print(f"TRL version: {trl.__version__}")

    # Force device selection based on availability
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cpu")
        print("Desired device not available. Falling back to CPU.")

    # Load tokenizer
    try:
        model_name = 'gpt2'  # Use 'gpt2' for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        tokenizer = None

    # Load model with Value Head and move to device
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        model = model.to(device)
        print(f"Model loaded and moved to {device}")
        print(f"Model is on device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    # Define PPO configuration without accelerator_kwargs
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        ppo_epochs=4
        # Removed accelerator_kwargs
    )

    # Create a reference model (frozen copy) and move to device
    try:
        ref_model = deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.to(device)
        print(f"Reference model is on device: {next(ref_model.parameters()).device}")
        print("Reference model created successfully")
    except Exception as e:
        print(f"Error creating reference model: {e}")
        ref_model = None

    # Initialize PPO trainer without passing 'device' directly
    try:
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,    # Explicitly pass the reference model
            tokenizer=tokenizer     # Explicitly pass the tokenizer
        )
        print("PPOTrainer initialized successfully")
        print(f"After PPOTrainer initialization, model is on device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"Error initializing PPOTrainer: {e}")
        return

    # Test PPOTrainer with dummy data
    dummy_query = torch.randint(0, 1000, (1, 50), device=device)
    dummy_response = torch.randint(0, 1000, (1, 50), device=device)
    dummy_reward = torch.tensor([0.5], dtype=torch.float32).to(device)  # Ensure shape [1]

    try:
        print("Running PPOTrainer.step() with dummy data...")
        ppo_trainer.step([dummy_query], [dummy_response], [dummy_reward])
        print("PPO step with dummy data completed successfully")
    except Exception as e:
        print(f"Error during PPOTrainer step with dummy data: {e}")

    # Prepare the dataset
    prompts = [
        "Describe a time when you had to make a difficult decision.",
        "How do you approach problem-solving in your daily life?",
        "What motivates you to achieve your goals?",
    ]
    data_loader = DataLoader(prompts, batch_size=1, shuffle=True)

    num_epochs = 1  # Adjust as needed

    # Training Loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            prompt = batch[0]
            print(f"Processing prompt: {prompt}")

            # Tokenize the prompt with fixed max_length and move to device
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=50,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
                assert inputs['input_ids'].shape[1] == 50, f"Prompt sequence length is {inputs['input_ids'].shape[1]}, expected 50"
            except Exception as e:
                print(f"Error tokenizing prompt: {e}")
                continue

            # Ensure model is on the correct device before generation
            model.to(device)
            print(f"Model is on device: {next(model.parameters()).device}")

            # Generate a response
            try:
                print("Generating response...")
                response_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id  # Ensure padding
                )
                print(f"response_ids.shape: {response_ids.shape}")  # Should be [1, 100]
                assert response_ids.shape[1] == 100, f"Total sequence length is {response_ids.shape[1]}, expected 100"
            except Exception as e:
                print(f"Error generating response: {e}")
                continue

            # Decode the response
            try:
                # Extract the response tokens, assuming they are the new tokens after the prompt
                response = tokenizer.decode(
                    response_ids[0][inputs['input_ids'].shape[-1]:],
                    skip_special_tokens=True
                )
                print(f"Generated response: {response}")
            except Exception as e:
                print(f"Error decoding response: {e}")
                response = ""

            # Compute the reward
            try:
                reward = compute_certainty(response)
                print(f"Computed certainty: {reward:.2f}")
            except Exception as e:
                print(f"Error computing certainty: {e}")
                reward = 0.0

            # Prepare PPO inputs
            query_tensors = inputs['input_ids']
            response_tensors = response_ids[:, inputs['input_ids'].shape[-1]:]

            # Verify tensor shapes
            print(f"query_tensors.shape: {query_tensors.shape}")        # Expected: [1, 50]
            print(f"response_tensors.shape: {response_tensors.shape}")  # Expected: [1, 50]
            assert query_tensors.shape[1] == 50, f"Query tensor sequence length is {query_tensors.shape[1]}, expected 50"
            assert response_tensors.shape[1] == 50, f"Response tensor sequence length is {response_tensors.shape[1]}, expected 50"

            # Before PPO step
            print(f"Before PPO step, model is on device: {next(model.parameters()).device}")

            # Run PPO step
            try:
                # Convert reward to a 1D tensor with shape [1]
                reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
                print(f"reward_tensor: {reward_tensor} (shape: {reward_tensor.shape})")
                assert reward_tensor.dim() == 1 and reward_tensor.shape[0] == 1, \
                    f"Reward tensor has invalid shape: {reward_tensor.shape}, expected [1]"

                # Debugging device assignments
                print(f"query_tensors device: {query_tensors.device}")
                print(f"response_tensors device: {response_tensors.device}")
                print(f"reward_tensor device: {reward_tensor.device}")

                # Run PPO step with batched reward tensor
                ppo_trainer.step([query_tensors], [response_tensors], [reward_tensor])
                print("PPO step completed successfully")
            except Exception as e:
                print(f"Error during PPOTrainer step: {e}")
                continue

            # After PPO step
            print(f"After PPO step, model is on device: {next(model.parameters()).device}")

            # Print progress
            print(f"Epoch {epoch}, Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Certainty: {reward:.2f}")
            print("---")

    # Save the fine-tuned model
    try:
        model.save_pretrained('fine_tuned_model')
        tokenizer.save_pretrained('fine_tuned_model')
        print("Model and tokenizer saved successfully")
    except Exception as e:
        print(f"Error saving model/tokenizer: {e}")

if __name__ == "__main__":
    main()