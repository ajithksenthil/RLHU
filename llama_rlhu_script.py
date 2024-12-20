import os
import sys
import torch
import torch.nn as nn
import warnings
import argparse
from typing import Optional
from transformers import (
    AutoTokenizer,  # Change this
    LlamaForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig
)
import bitsandbytes as bnb
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import Dataset

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import your PsychologicalProfiler
from user_profile_project.psych_profiler_1_copy import PsychologicalProfiler


# Initialize the profiler
profiler = PsychologicalProfiler()

def compute_certainty(response: str, context: str = "general") -> float:
    profile = profiler.update_profile(response, context)
    certainty = profile.get('certainty', 0.0)
    certainty = max(0.0, min(certainty, 0.99))
    return certainty

def parse_args():
    parser = argparse.ArgumentParser(description="RLHU Training Script for Llama 3.2")
    parser.add_argument('--device', type=str, default='mps', help="Device to use: 'cuda', 'cpu', 'mps'")
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B', help="Llama model name or path")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    parser.add_argument('--use_4bit', action='store_true', help="Use 4-bit quantization")
    parser.add_argument('--use_8bit', action='store_true', help="Use 8-bit quantization")
    return parser.parse_args()

class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=32):
        self.input_ids_list = []
        self.attention_mask_list = []
        self.queries = []
        
        for prompt in prompts:
            encodings = tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            self.input_ids_list.append(encodings['input_ids'][0])
            self.attention_mask_list.append(encodings['attention_mask'][0])
            self.queries.append(prompt)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_mask_list[idx],
            "response_ids": self.input_ids_list[idx].clone(),  # Required by trl
            "response": self.queries[idx]
        }

class CustomAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    base_model_prefix = 'pretrained_model'
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        kwargs["output_hidden_states"] = True
        
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states[-1]
        logits = self.pretrained_model.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)
        
        return {
            "logits": logits,
            "value": value,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None
        }

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def score(self, hidden_states):
        return self.v_head(hidden_states).squeeze(-1)

class CustomPPOTrainer(PPOTrainer):
    def compute_rewards(self, samples, **kwargs):
        responses = samples["response"]
        rewards = []
        for response in responses:
            reward = compute_certainty(response)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.accelerator.device)
        rewards = rewards.unsqueeze(1).expand(-1, samples['response_tokens'].shape[1])
        return rewards

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not hasattr(self.accelerator.unwrap_model(self.model), 'save_pretrained'):
            raise ValueError("Trainer.model does not have a save_pretrained method")
        
        # Unwrap and save
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

# Define a custom dummy model
class DummyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size, seq_length = input_ids.size()
        hidden_size = self.config.hidden_size
        device = input_ids.device
        # Create dummy hidden states
        hidden_states = torch.zeros(batch_size, seq_length, hidden_size, device=device)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=(hidden_states,)
        )

# Define a dummy reward model
class DummyRewardModel(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = 'dummy_model'

    def __init__(self, config):
        super().__init__(config)
        self.dummy_model = DummyModel(config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.dummy_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def score(self, hidden_states):
        # Return a dummy score tensor
        batch_size, seq_length, hidden_size = hidden_states.size()
        device = hidden_states.device
        # Return zeros of shape (batch_size, seq_length)
        return torch.zeros(batch_size, seq_length, device=device)


def main():
    args = parse_args()

    # Device setup for Mac
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        # Enable async implementation
        torch.backends.mps.enable_async = True
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using {device} device for training.")
    
    # Memory management for Mac
    if device.type == "mps":
        # Clear GPU memory
        torch.mps.empty_cache()
        # Set memory limit
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

    # Initialize tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # Initialize policy and reference models
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float32,  # Use float32 for MPS
        "low_cpu_mem_usage": True,
    }

    # Initialize policy and reference models
    # Initialize policy and reference models
    # Initialize policy and reference models using the custom class
    policy = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.float32
    ).to(device)
    
    ref_model = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.float32
    ).to(device)

    # Ensure return_dict is set at multiple levels
    policy.pretrained_model.config.return_dict = True
    policy.config.return_dict = True
    policy.config.output_hidden_states = True

    ref_model.pretrained_model.config.return_dict = True
    ref_model.config.return_dict = True
    ref_model.config.output_hidden_states = True

    # Initialize dummy reward model
    dummy_config = PretrainedConfig(hidden_size=policy.config.hidden_size)
    reward_model = DummyRewardModel(dummy_config).to(device)

    # Ensure models are in training mode
    policy.train()
    ref_model.train()
    

    # Configuration setup
    for model in [policy, ref_model]:
        model.pretrained_model.config.return_dict = True
        model.config.return_dict = True
        model.config.output_hidden_states = True

    # Generation config
    gen_config = GenerationConfig.from_model_config(policy.config)
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.return_dict_in_generate = True
    policy.generation_config = gen_config
    ref_model.generation_config = gen_config
    

   
    # PPO configuration
    # Optimized PPO config for Mac
    # PPO Configuration
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=4,  # Reduced for MPS device
        batch_size=8,
        gradient_checkpointing=True,
        target_kl=0.1,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
        max_grad_norm=1.0,
        seed=42,
        output_dir="./ppo_results",
        gradient_accumulation_steps=1,
        remove_unused_columns=False,  # Important for custom datasets
        log_with=None,  # Disable logging for now
        torch_dtype=torch.float32,  # Explicitly set dtype for MPS
        use_cache=False  # Disable caching for more stable training
    )

    # Example prompts
    prompts = [
        "Describe a time when you had to make a difficult decision.",
        "How do you approach problem-solving in your daily life?",
        "What motivates you to achieve your goals?",
        "What is your favorite hobby and why?",
        "Tell me about a memorable experience you had.",
    ]

    # Dataset preparation
    train_dataset = PromptDataset(prompts, tokenizer, max_length=32)  # Reduced max_length for testing
    eval_dataset = train_dataset

    # Initialize the dummy reward model
    dummy_config = PretrainedConfig(hidden_size=policy.config.hidden_size)
    reward_model = DummyRewardModel(dummy_config).to(device)

    print("Initializing PPO trainer...")
    # Initialize PPO trainer with minimal required arguments
    # Initialize PPO trainer with correct parameter names
    # Initialize PPO trainer with all required components
    # Initialize PPO trainer with correct parameters
    ppo_trainer = CustomPPOTrainer(
        config=ppo_config,
        model=policy,               # Use 'model' as that's the correct parameter name
        tokenizer=tokenizer,
        dataset=train_dataset       # Use 'dataset' instead of 'train_dataset'
    )

    # Add error handling for training
    try:
        print("Starting training...")
        for epoch in range(ppo_config.num_train_epochs):
            print(f"Starting epoch {epoch + 1}/{ppo_config.num_train_epochs}")
            ppo_trainer.train()
            print(f"Completed epoch {epoch + 1}")
    except Exception as e:
        print(f"Error during training: {e}")
        print(f"Error type: {type(e)}")  # Added for better debugging
        print(f"Error args: {e.args}")    # Added for better debugging
        raise e
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Save the trained model
    try:
        output_dir = 'llama_fine_tuned_model'
        ppo_trainer.save_model(output_dir)
        print(f"Model saved successfully to {output_dir}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
