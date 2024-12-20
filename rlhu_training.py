# rlhu_training.py

import os

# Disable CUDA devices if any and enable MPS fallback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from transformers import (
    GPT2Tokenizer,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)
from torch.utils.data import Dataset
import sys
import argparse
import warnings

# **Add this import statement**
from typing import Optional

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
    profile = profiler.update_profile(response, context)
    certainty = profile.get('certainty', 0.0)
    certainty = max(0.0, min(certainty, 0.99))
    return certainty

def parse_args():
    parser = argparse.ArgumentParser(description="RLHU Training Script")
    parser.add_argument('--device', type=str, default='mps', help="Device to use: 'cuda', 'cpu', 'mps'")
    return parser.parse_args()

# Custom Dataset Class
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=32):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
        }

# Define the custom policy model with modified forward method and score method
class CustomAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    base_model_prefix = "pretrained_model"

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Ensure output_hidden_states is True
        kwargs["output_hidden_states"] = True
        # Get outputs from the pretrained model
        transformer_outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Access the last hidden state via hidden_states[-1]
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # Compute logits and values
        logits = self.pretrained_model.lm_head(last_hidden_state)
        values = self.v_head(last_hidden_state).squeeze(-1)

        # Create a ModelOutput object with logits and values
        outputs = CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        outputs["values"] = values
        return outputs

    def score(self, hidden_states):
        return self.v_head(hidden_states).squeeze(-1)

# Subclass PPOTrainer and override compute_rewards and save_model
from trl import PPOTrainer

class CustomPPOTrainer(PPOTrainer):
    def compute_rewards(self, samples, **kwargs):
        # (Your existing compute_rewards code)
        responses = samples["response"]
        rewards = []
        for response in responses:
            reward = compute_certainty(response)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.accelerator.device)
        rewards = rewards.unsqueeze(1).expand(-1, samples['response_tokens'].shape[1])
        return rewards

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # Adjusted save_model method
        output_dir = output_dir if output_dir is not None else self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not hasattr(self.policy, 'save_pretrained'):
            raise ValueError("Trainer.policy does not have a save_pretrained method")
        self.policy.save_pretrained(output_dir, safe_serialization=False)
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

    # Force device selection based on availability
    global device  # Declare device as global so it can be accessed in compute_certainty
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cpu")
        print("Desired device not available. Falling back to CPU.")

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize policy and reference models using the custom class
    policy = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, return_dict=True
    ).to(device)
    ref_model = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, return_dict=True
    ).to(device)

    # Ensure return_dict is set at multiple levels
    policy.pretrained_model.config.return_dict = True
    policy.config.return_dict = True
    policy.config.output_hidden_states = True

    ref_model.pretrained_model.config.return_dict = True
    ref_model.config.return_dict = True
    ref_model.config.output_hidden_states = True

    # For generation, we set return_dict_in_generate=True
    gen_config = GenerationConfig.from_model_config(policy.config)
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.return_dict_in_generate = True
    policy.generation_config = gen_config
    ref_model.generation_config = gen_config

    # Define PPO configuration
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        output_dir="./ppo_results",
        per_device_train_batch_size=8,  # Adjust based on your hardware
        num_mini_batches=1,
        gamma=0.99,
        lam=0.95,
        whiten_rewards=True,
        response_length=20,
    )

    # Prepare the dataset
    prompts = [
        "Describe a time when you had to make a difficult decision.",
        "How do you approach problem-solving in your daily life?",
        "What motivates you to achieve your goals.",
        "What is your favorite hobby and why?",
        "Tell me about a memorable experience you had.",
        "How do you handle stress and pressure?",
        "What are your long-term career goals?",
        "Describe an accomplishment you're proud of.",
        "How do you stay motivated during challenging times?",
    ]
    train_dataset = PromptDataset(prompts, tokenizer)
    eval_dataset = train_dataset  # Use the same dataset for evaluation

    # Initialize the dummy reward model
    dummy_config = PretrainedConfig(hidden_size=policy.config.hidden_size)
    reward_model = DummyRewardModel(dummy_config).to(device)

    print("Initializing PPO trainer...")
    # PPOTrainer initialization
    ppo_trainer = CustomPPOTrainer(
        config=ppo_config,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_model,
        reward_model=reward_model,  # Pass the dummy reward model here
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        value_model=policy,
    )

    print("Starting training...")
    ppo_trainer.train()
    print("Training completed.")

    # Save the fine-tuned model
    try:
        policy.save_pretrained('fine_tuned_model', safe_serialization=False)
        tokenizer.save_pretrained('fine_tuned_model')
        print("Model and tokenizer saved successfully")
    except Exception as e:
        print(f"Error saving model/tokenizer: {e}")

if __name__ == "__main__":
    main()
