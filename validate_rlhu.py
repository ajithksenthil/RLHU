import os
import json
import torch
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from rlhu_training import (
    CustomAutoModelForCausalLMWithValueHead,
    PsychologicalProfiler,
    compute_certainty
)

def load_model_and_tokenizer(model_path, device):
    """Load the fine-tuned model and tokenizer."""
    try:
        print(f"Loading model and tokenizer from {model_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            return_dict=True
        ).to(device)
        
        # Ensure model is in evaluation mode
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model and tokenizer: {str(e)}")
        # If model_path doesn't exist, try loading base GPT2
        print("Attempting to load base GPT2 model...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            
            model = CustomAutoModelForCausalLMWithValueHead.from_pretrained(
                'gpt2',
                return_dict=True
            ).to(device)
            
            model.eval()
            return model, tokenizer
        except Exception as e2:
            raise Exception(f"Failed to load both fine-tuned and base models: {str(e2)}")

def measure_personality_certainty(conversation_segment, true_personality_type):
    """
    Measure how certain the profiler is about the true personality type
    based on the conversation up to this point.
    """
    profiler = PsychologicalProfiler()
    conversation_text = "\n".join([
        f"{msg['speaker']}: {msg['message']}" 
        for msg in conversation_segment
    ])
    # Update the profile with the conversation text
    profile = profiler.update_profile(conversation_text, context=true_personality_type)
    # Get the certainty from the profile for the true personality type
    certainty = profile.get('certainty', 0.0)
    return certainty

def generate_interviewer_question(model, tokenizer, conversation_history, personality_type, device):
    """Generate next interviewer question optimized for personality detection."""
    try:
        prompt = (f"Previous conversation about a person with personality type {personality_type}:\n\n")
        for msg in conversation_history:
            prompt += f"{msg['speaker']}: {msg['message']}\n"
        prompt += "\nGenerate the next interviewer question to better understand their personality type:"
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():  # Add this for inference
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True).split(":")[-1].strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

def evaluate_conversation_certainty(model, tokenizer, conversation, device):
    """
    Evaluate how certainty about the true personality type evolves through the conversation,
    comparing original questions vs. generated questions.
    """
    true_personality = conversation['personality_type']
    results = {
        'id': conversation['id'],
        'personality_type': true_personality,
        'original_certainty_progression': [],
        'generated_questions': [],
        'generated_certainty_progression': []
    }
    
    # Measure original certainty progression
    current_messages = []
    for msg in conversation['messages']:
        current_messages.append(msg)
        if msg['speaker'] == 'target':
            certainty = measure_personality_certainty(current_messages, true_personality)
            results['original_certainty_progression'].append({
                'position': len(current_messages) - 1,
                'certainty': certainty
            })
    
    # Generate alternative questions and measure their impact
    conversation_history = []
    generated_messages = []
    for msg in conversation['messages']:
        if msg['speaker'] == 'interviewer':
            generated_question = generate_interviewer_question(
                model, tokenizer, conversation_history, 
                true_personality, device
            )
            results['generated_questions'].append({
                'position': len(conversation_history),
                'original': msg['message'],
                'generated': generated_question
            })
            generated_messages.append({
                'speaker': 'interviewer',
                'message': generated_question
            })
        else:
            generated_messages.append(msg)
            if msg['speaker'] == 'target':
                certainty = measure_personality_certainty(generated_messages, true_personality)
                results['generated_certainty_progression'].append({
                    'position': len(generated_messages) - 1,
                    'certainty': certainty
                })
        conversation_history.append(msg)
    
    # Calculate certainty improvement metrics
    if results['original_certainty_progression'] and results['generated_certainty_progression']:
        results['certainty_metrics'] = {
            'original_final_certainty': results['original_certainty_progression'][-1]['certainty'],
            'generated_final_certainty': results['generated_certainty_progression'][-1]['certainty'],
            'original_avg_certainty': np.mean([p['certainty'] for p in results['original_certainty_progression']]),
            'generated_avg_certainty': np.mean([p['certainty'] for p in results['generated_certainty_progression']]),
            'certainty_improvement': (
                results['generated_certainty_progression'][-1]['certainty'] -
                results['original_certainty_progression'][-1]['certainty']
            )
        }
    
    return results

def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    try:
        # Load model, tokenizer, and conversations
        model, tokenizer = load_model_and_tokenizer(args.model_path, device)
        conversations = load_conversations(args.conversations_path)
        
        # Evaluate all conversations
        validation_results = []
        for conversation in conversations:
            try:
                results = evaluate_conversation_certainty(model, tokenizer, conversation, device)
                validation_results.append(results)
                print(f"Evaluated conversation {results['id']}")
            except Exception as e:
                print(f"Error evaluating conversation {conversation['id']}: {str(e)}")
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'total_conversations': len(validation_results),
            'average_certainty_improvement': np.mean([
                r['certainty_metrics']['certainty_improvement']
                for r in validation_results
                if 'certainty_metrics' in r
            ]),
            'average_original_final_certainty': np.mean([
                r['certainty_metrics']['original_final_certainty']
                for r in validation_results
                if 'certainty_metrics' in r
            ]),
            'average_generated_final_certainty': np.mean([
                r['certainty_metrics']['generated_final_certainty']
                for r in validation_results
                if 'certainty_metrics' in r
            ])
        }
        
        # Save results
        output = {
            'validation_results': validation_results,
            'aggregate_metrics': aggregate_metrics
        }
        
        with open(args.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\nPersonality Type Certainty Results:")
        print(f"Total conversations analyzed: {aggregate_metrics['total_conversations']}")
        print(f"Average certainty improvement: {aggregate_metrics['average_certainty_improvement']:.3f}")
        print(f"Average original final certainty: {aggregate_metrics['average_original_final_certainty']:.3f}")
        print(f"Average generated final certainty: {aggregate_metrics['average_generated_final_certainty']:.3f}")
        print(f"\nDetailed results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="RLHU Personality Type Certainty Validation")
    parser.add_argument('--model_path', type=str, default='fine_tuned_model',
                       help="Path to the fine-tuned model")
    parser.add_argument('--conversations_path', type=str, default='synthetic_conversations.json',
                       help="Path to the synthetic conversations JSON file")
    parser.add_argument('--device', type=str, default='mps',
                       help="Device to use: 'cuda', 'cpu', or 'mps'")
    parser.add_argument('--output_path', type=str, default='personality_certainty_results.json',
                       help="Path to save validation results")
    return parser.parse_args()

def load_conversations(file_path):
    """Load synthetic conversations from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading conversations: {str(e)}")

if __name__ == "__main__":
    main()