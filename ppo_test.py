import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GenerationConfig,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)

#######################################
# Setup
#######################################

# Device setup with MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class DummyDataset(Dataset):
    def __init__(self, tokenizer, size=10):
        self.tokenizer = tokenizer
        self.size = size
        self.query = "This morning I went to the "

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
        }


# Define a custom reward model based on GPT-2
class CustomRewardModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # For simplicity, we'll use the pretrained weights

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Return a constant reward of ones, with required outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        # Create a mock reward tensor
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        # Return a tensor of ones as the last hidden state to simulate rewards
        last_hidden_state = torch.ones(batch_size, sequence_length, self.config.hidden_size).to(input_ids.device)
        outputs.hidden_states = outputs.hidden_states + (last_hidden_state,)
        return outputs

    def score(self, hidden_states):
        # Return a constant score
        # Simulate scoring by returning a tensor of ones
        batch_size = hidden_states.size(0)
        sequence_length = hidden_states.size(1)
        return torch.ones(batch_size, sequence_length).to(hidden_states.device)


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


model_name = "gpt2"

# Initialize models using the custom class
policy = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
    model_name, return_dict=True
).to(device)

ref_model = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
    model_name, return_dict=True
).to(device)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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
gen_config.return_dict_in_generate = True  # Set this True for generation
policy.generation_config = gen_config
ref_model.generation_config = gen_config

# Updated configuration with only supported parameters
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    output_dir="./ppo_results",
    per_device_train_batch_size=8,
    num_mini_batches=2,
    gamma=0.99,
    lam=0.95,
    whiten_rewards=True,
    response_length=20,
    # Uncomment the line below to disable evaluation
    # eval_steps=0,
)

# Initialize the custom reward model
reward_model = CustomRewardModel.from_pretrained(model_name).to(device)

train_dataset = DummyDataset(tokenizer, size=32)
eval_dataset = train_dataset  # Use the same dataset for evaluation

print("Initializing PPO trainer...")
# PPOTrainer initialization
ppo_trainer = PPOTrainer(
    config=ppo_config,
    processing_class=tokenizer,
    policy=policy,
    ref_policy=ref_model,
    reward_model=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add this line
    value_model=policy,  # Use policy as the value model
)

print("Setting up batch data...")
# Get a batch of data
data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(
    train_dataset,
    batch_size=ppo_config.per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

try:
    # Get batch and prepare tensors
    batch = next(iter(dataloader))
    query_tensors = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    print("Starting training...")
    # Train the model
    ppo_trainer.train()

    print("\nGenerating sample responses...")
    # Generate sample responses
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 20,
        "return_dict_in_generate": True,
    }

    response_tensors = policy.generate(
        input_ids=query_tensors,
        attention_mask=attention_mask,
        **generation_kwargs,
    )

    # Print sample generations
    queries = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
    responses = tokenizer.batch_decode(
        response_tensors.sequences, skip_special_tokens=True
    )

    print("\nSample generations:")
    for query, response in zip(queries[:2], responses[:2]):
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

except Exception as e:
    print(f"Error during execution: {e}")
    print(f"Query tensors device: {query_tensors.device}")
    print(f"Model device: {policy.pretrained_model.device}")
    print(f"Attention mask device: {attention_mask.device}")
    # Print model output format for debugging
    with torch.no_grad():
        output = policy(query_tensors)
        print(f"\nModel output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Tuple length: {len(output)}")
            print(f"Tuple contents: {[type(x) for x in output]}")
    raise
