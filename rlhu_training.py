# rlhu_training.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from torch.utils.data import DataLoader
import sys 
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# import your PsychologicalProfiler
from user_profile_project.psych_profiler import PsychologicalProfiler

# Initialize the profiler
profiler = PsychologicalProfiler()

# Define the compute_certainty function
def compute_certainty(response: str, context: str = "general") -> float:
    _, certainty = profiler.update_profile(response, context)
    certainty = max(0.0, min(certainty, 0.99))
    return certainty

def main():
    # Choose a model
    model_name = 'EleutherAI/gpt-neo-1.3B'  # Replace with your chosen model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define PPO configuration
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=1,
        ppo_epochs=4,
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(model, tokenizer, **ppo_config)

    # Prepare the dataset
    prompts = [
        "Describe a time when you had to make a difficult decision.",
        "How do you approach problem-solving in your daily life?",
        "What motivates you to achieve your goals?",
        # Add more prompts as needed
    ]
    data_loader = DataLoader(prompts, batch_size=1, shuffle=True)

    num_epochs = 1  # Adjust as needed

    # Training Loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            prompt = batch[0]

            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

            # Generate a response
            response_ids = model.generate(
                **inputs,
                max_length=tokenizer.model_max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            # Decode the response
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Compute the reward
            reward = compute_certainty(response)

            # Prepare PPO inputs
            query_tensors = inputs['input_ids']
            response_tensors = response_ids[:, inputs['input_ids'].shape[-1]:]

            # Run PPO step
            ppo_trainer.step(query_tensors, response_tensors, [reward])

            # Optional: Print progress
            print(f"Epoch {epoch}, Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Certainty: {reward:.2f}")
            print("---")

    # Save the fine-tuned model
    model.save_pretrained('fine_tuned_model')
    tokenizer.save_pretrained('fine_tuned_model')

if __name__ == "__main__":
    main()