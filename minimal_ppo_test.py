# minimal_ppo_test_v2.py

import os
import torch
from trl import PPOv2Trainer, PPOv2Config, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from copy import deepcopy

def main():
    # Set environment variables before importing torch or other libraries
    os.environ['CUDA_VISIBLE_DEVICES'] = ''          # Disable CUDA
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'   # Enable MPS fallback
    os.environ["TOKENIZERS_PARALLELISM"] = "false"     # Suppress tokenizers parallelism warnings

    # Force device to CPU to avoid MPS limitations for this test
    device = torch.device("cpu")
    print("Using CPU device for training.")

    # Initialize tokenizer and model
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        ref_model = deepcopy(model).eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.to(device)
        print("Reference model created successfully")
    except Exception as e:
        print(f"Error creating reference model: {e}")
        return

    # Configure PPO
    ppo_config = PPOv2Config(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        ppo_epochs=4
    )

    try:
        ppo_trainer = PPOv2Trainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer
        )
        print("PPOv2Trainer initialized successfully")
    except Exception as e:
        print(f"Error initializing PPOv2Trainer: {e}")
        return

    # Create dummy data
    dummy_query = torch.randint(0, 1000, (1, 50), device=device)
    dummy_response = torch.randint(0, 1000, (1, 50), device=device)
    dummy_reward = torch.tensor([0.5], dtype=torch.float32).to(device)  # Shape: [1]

    # Debugging tensor details
    print(f"dummy_query: {dummy_query}")
    print(f"dummy_response: {dummy_response}")
    print(f"dummy_reward: {dummy_reward}")

    # Perform PPO step with dummy data
    try:
        print("Running PPOv2Trainer.step() with dummy data...")
        ppo_trainer.step([dummy_query], [dummy_response], [dummy_reward])
        print("PPO step with dummy data completed successfully")
    except Exception as e:
        print(f"Error during PPOv2Trainer step with dummy data: {e}")

if __name__ == "__main__":
    main()