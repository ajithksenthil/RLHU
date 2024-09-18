import os
import datetime
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
import json
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
import math

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

# Constants
PAIRINGS = [
    "Ti Hero - Fe Inferior", "Fe Hero - Ti Inferior",
    "Te Hero - Fi Inferior", "Fi Hero - Te Inferior",
    "Ne Hero - Si Inferior", "Si Hero - Ne Inferior",
    "Ni Hero - Se Inferior", "Se Hero - Ni Inferior"
]

COGNITIVE_FUNCTIONS = ['Ti', 'Te', 'Fi', 'Fe', 'Ni', 'Ne', 'Si', 'Se']

class PsychologicalProfiler:
    def __init__(self):
        """
        Initialize the PsychologicalProfiler with necessary components and parameters.
        """
        # Initialize OpenAI LLM
        self.llm = OpenAI(temperature=0.7)

        # Initialize prompts for the three-step process
        self.function_identification_prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""
            Analyze the following conversation and assess the presence and strength of each cognitive function (Ti, Te, Fi, Fe, Ni, Ne, Si, Se). Provide a brief analysis for all functions, even if they're not strongly present.

            Conversation: {conversation}

            For each function, provide:
            1. The function name (Ti, Te, Fi, Fe, Ni, Ne, Si, Se)
            2. A rating from 0-10 (0 being not present at all, 10 being very strongly present)
            3. A brief explanation of how it's being used or why it's not evident (1-2 sentences max)

            Example format:
            Ti: 7 - Clear logical analysis present in the argument about X.
            Te: 3 - Some organization of external information, but not a primary focus.
            ...

            Provide an analysis for all 8 functions in this format.
            """
        )

        self.pairing_analysis_prompt = PromptTemplate(
            input_variables=["function_analysis", "conversation"],
            template="""
            Based on the following function analysis and the original conversation, determine the most likely Hero-Inferior pairing. Consider all possible pairings and their associated strengths and challenges:

            Ti Hero - Fe Inferior:
            Strengths: Logical analysis, problem-solving, objective decision-making
            Challenges: Emotional expression, social harmony, empathy in relationships

            Fe Hero - Ti Inferior:
            Strengths: Social awareness, empathy, creating harmony in groups
            Challenges: Logical analysis, maintaining objectivity, individual decision-making

            Te Hero - Fi Inferior:
            Strengths: Efficient organization, strategic planning, objective goal-setting
            Challenges: Personal value alignment, emotional self-awareness, moral decision-making

            Fi Hero - Te Inferior:
            Strengths: Strong personal values, authenticity, moral integrity
            Challenges: External organization, efficiency in systems, objective goal-setting

            Ne Hero - Si Inferior:
            Strengths: Idea generation, seeing possibilities, abstract connections
            Challenges: Detailed memory recall, maintaining routines, practical application of past experiences

            Si Hero - Ne Inferior:
            Strengths: Detailed memory, establishing routines, practical application of past experiences
            Challenges: Generating new ideas, seeing abstract possibilities, adapting to change

            Ni Hero - Se Inferior:
            Strengths: Long-term vision, pattern recognition, strategic foresight
            Challenges: Present moment awareness, sensory engagement, adapting to immediate environment

            Se Hero - Ni Inferior:
            Strengths: Present moment awareness, sensory engagement, quick adaptation to environment
            Challenges: Long-term vision, recognizing abstract patterns, strategic planning

            Original Conversation:
            {conversation}

            Function Analysis:
            {function_analysis}

            Provide:
            1. The most likely Hero-Inferior pairing
            2. A brief explanation for why this pairing is most likely, referencing the strengths and challenges, and citing specific examples from the conversation (2-3 sentences max)

            Present your analysis in a clear, concise format.
            """
        )

        self.structured_output_prompt = PromptTemplate(
            input_variables=["pairing_analysis"],
            template="""
            Based on the following pairing analysis, create a structured output summarizing the key findings.

            Pairing Analysis: {pairing_analysis}

            Generate a response in the following format:
            FUNCTION_RATINGS_START
            Ti: [rating] - [evidence]
            Te: [rating] - [evidence]
            Fi: [rating] - [evidence]
            Fe: [rating] - [evidence]
            Ni: [rating] - [evidence]
            Ne: [rating] - [evidence]
            Si: [rating] - [evidence]
            Se: [rating] - [evidence]
            FUNCTION_RATINGS_END
            PRIMARY_PAIRING: [Hero function] Hero - [Inferior function] Inferior
            PAIRING_EXPLANATION: [Brief explanation]
            SECONDARY_PAIRING: [Hero function] Hero - [Inferior function] Inferior
            SECONDARY_EXPLANATION: [Brief explanation]

            Ensure all sections are filled out. Use 'None' if no secondary pairing is identified.
            """
        )

        # Initialize other components
        self.pairing_counts = Counter()
        self.total_assessments = 0
        self.conversation_data = []

    
    def identify_functions_with_reasoning(self, conversation: str) -> str:
        prompt = PromptTemplate(
            input_variables=["conversation"],
            template="""
            Analyze the following conversation and assess the presence and strength of each cognitive function (Ti, Te, Fi, Fe, Ni, Ne, Si, Se). Provide detailed reasoning and evidence for each function.

            Conversation: {conversation}

            For each function, consider:
            1. How strongly is it exhibited?
            2. What specific examples in the conversation demonstrate this function?
            3. How does this function interact with other apparent functions?

            Provide your analysis in a clear, detailed format without using any JSON structure.
            """
        )
        return self.llm(prompt.format(conversation=conversation))
    
    def analyze_pairings_with_reasoning(self, function_analysis: str, conversation: str) -> str:
        return self.llm(self.pairing_analysis_prompt.format(
            function_analysis=function_analysis,
            conversation=conversation
        ))
    
    def generate_structured_output(self, pairing_analysis: str) -> Dict:
        prompt = self.structured_output_prompt.format(pairing_analysis=pairing_analysis)
        
        try:
            output = self.llm.invoke(prompt)
            return self.parse_structured_output(output)
        except Exception as e:
            print(f"Error in generate_structured_output: {e}")
            print("Raw output:", output)
            return self._get_default_json_structure()

    def parse_structured_output(self, output: str) -> Dict:
        lines = output.split('\n')
        result = {
            "function_ratings": [],
            "primary_pairing": "",
            "pairing_explanation": "",
            "secondary_pairing": None,
            "secondary_explanation": None
        }
        
        in_ratings = False
        for line in lines:
            line = line.strip()
            if line == "FUNCTION_RATINGS_START":
                in_ratings = True
                continue
            elif line == "FUNCTION_RATINGS_END":
                in_ratings = False
                continue
            
            if in_ratings:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    func, rest = parts
                    rest_parts = rest.split('-', 1)
                    if len(rest_parts) == 2:
                        rating, evidence = rest_parts
                    else:
                        rating, evidence = rest_parts[0], ""
                    
                    try:
                        rating_value = int(rating.strip())
                    except ValueError:
                        rating_value = 0  # Default to 0 if rating is not a valid integer
                    
                    result["function_ratings"].append({
                        "function": func.strip(),
                        "rating": rating_value,
                        "evidence": evidence.strip()
                    })
            elif line.startswith("PRIMARY_PAIRING:"):
                result["primary_pairing"] = line.split(':', 1)[1].strip()
            elif line.startswith("PAIRING_EXPLANATION:"):
                result["pairing_explanation"] = line.split(':', 1)[1].strip()
            elif line.startswith("SECONDARY_PAIRING:"):
                result["secondary_pairing"] = line.split(':', 1)[1].strip()
            elif line.startswith("SECONDARY_EXPLANATION:"):
                result["secondary_explanation"] = line.split(':', 1)[1].strip()
        
        # Ensure all 8 functions are present
        existing_functions = {r['function'] for r in result['function_ratings']}
        for func in COGNITIVE_FUNCTIONS:
            if func not in existing_functions:
                result['function_ratings'].append({
                    "function": func,
                    "rating": 0,
                    "evidence": "No evidence provided"
                })
        
        return result
        
    def _validate_json_structure(self, parsed_json: Dict):
        """Validate the structure of the parsed JSON."""
        assert 'function_ratings' in parsed_json, "Missing 'function_ratings'"
        assert len(parsed_json['function_ratings']) == 8, "Incorrect number of function ratings"
        
        for rating in parsed_json['function_ratings']:
            assert all(key in rating for key in ['function', 'rating', 'evidence']), "Missing keys in function rating"
            assert isinstance(rating['rating'], int) and 0 <= rating['rating'] <= 10, f"Invalid rating for {rating['function']}"
        
        assert 'primary_pairing' in parsed_json, "Missing 'primary_pairing'"
        assert 'pairing_explanation' in parsed_json, "Missing 'pairing_explanation'"
        assert 'secondary_pairing' in parsed_json, "Missing 'secondary_pairing'"
        assert 'secondary_explanation' in parsed_json, "Missing 'secondary_explanation'"

    def _get_default_json_structure(self) -> Dict:
        """Return a default JSON structure."""
        return {
            "function_ratings": [
                {"function": func, "rating": 0, "evidence": "No data available"}
                for func in ['Ti', 'Te', 'Fi', 'Fe', 'Ni', 'Ne', 'Si', 'Se']
            ],
            "primary_pairing": "Unknown Hero - Unknown Inferior",
            "pairing_explanation": "No data available",
            "secondary_pairing": None,
            "secondary_explanation": None
        }


    def update_profile(self, new_message: str, context: str = "general") -> Tuple[str, float]:
        try:
            # Step 1: Function Identification with Reasoning
            function_analysis = self.identify_functions_with_reasoning(new_message)
            
            # Step 2: Pairing Analysis with Reasoning
            pairing_analysis = self.analyze_pairings_with_reasoning(function_analysis, new_message)
            
            # Step 3: Generate Structured Output
            structured_output = self.generate_structured_output(pairing_analysis)
            
            # Extract the most likely pairing
            most_likely_pairing = structured_output.get('primary_pairing', 'Unknown - Unknown')
            
            # Update counts
            self.pairing_counts[most_likely_pairing] += 1
            self.total_assessments += 1
            
            # Store conversation data
            self.conversation_data.append({
                'timestamp': datetime.datetime.now(),
                'message': new_message,
                'context': context,
                'most_likely_pairing': most_likely_pairing,
                'analysis': structured_output
            })
            
            # Generate profile and calculate certainty
            profile = self.generate_profile()
            certainty = self.calculate_certainty()
            
            return profile, certainty
        except Exception as e:
            print(f"Error in update_profile: {e}")
            print("Function analysis:", function_analysis)
            print("Pairing analysis:", pairing_analysis)
            print("Structured output:", structured_output)
            raise
        
    
    
    # Multiplicative certainty calculation
    def calculate_certainty(self) -> float:
       if not self.total_assessments:
           return 0.0
       
       probabilities = [count / self.total_assessments for count in self.pairing_counts.values()]
       
       # Calculate entropy
       entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
       
       # Max entropy is log2(8) for 8 possible pairings
       max_entropy = math.log2(8)
       
       # Normalize entropy to [0, 1] range and invert
       normalized_entropy = 1 - (entropy / max_entropy)
       
       # Factor in the amount of data
       data_factor = 1 - math.exp(-self.total_assessments / 10)  # Asymptotically approaches 1
       
       certainty = normalized_entropy * data_factor
       
       return certainty
            
    

    def generate_profile(self) -> str:
        if not self.total_assessments:
            return "Not enough data for profiling"
        
        most_common_pairing = self.pairing_counts.most_common(1)[0][0]
        probability = self.pairing_counts[most_common_pairing] / self.total_assessments
        
        profile = f"Most likely Hero-Inferior pairing: {most_common_pairing}\n"
        profile += f"Probability: {probability:.2f}\n"
        profile += f"Total assessments: {self.total_assessments}\n"
        profile += "Potential strengths and challenges:\n"
        profile += self.get_strengths_and_challenges(most_common_pairing)
        return profile


    def get_strengths_and_challenges(self, pairing: str) -> str:
        """
        Get the strengths and challenges associated with a specific Hero-Inferior pairing.

        Parameters:
        - pairing (str): The Hero-Inferior pairing.

        Returns:
        - str: A string describing the strengths and challenges of the pairing.
        """
        strengths_challenges = {
            "Ti Hero - Fe Inferior": {
                "strengths": "Logical analysis, problem-solving, objective decision-making",
                "challenges": "Emotional expression, social harmony, empathy in relationships"
            },
            "Fe Hero - Ti Inferior": {
                "strengths": "Social awareness, empathy, creating harmony in groups",
                "challenges": "Logical analysis, maintaining objectivity, individual decision-making"
            },
            "Te Hero - Fi Inferior": {
                "strengths": "Efficient organization, strategic planning, objective goal-setting",
                "challenges": "Personal value alignment, emotional self-awareness, moral decision-making"
            },
            "Fi Hero - Te Inferior": {
                "strengths": "Strong personal values, authenticity, moral integrity",
                "challenges": "External organization, efficiency in systems, objective goal-setting"
            },
            "Ne Hero - Si Inferior": {
                "strengths": "Idea generation, seeing possibilities, abstract connections",
                "challenges": "Detailed memory recall, maintaining routines, practical application of past experiences"
            },
            "Si Hero - Ne Inferior": {
                "strengths": "Detailed memory, establishing routines, practical application of past experiences",
                "challenges": "Generating new ideas, seeing abstract possibilities, adapting to change"
            },
            "Ni Hero - Se Inferior": {
                "strengths": "Long-term vision, pattern recognition, strategic foresight",
                "challenges": "Present moment awareness, sensory engagement, adapting to immediate environment"
            },
            "Se Hero - Ni Inferior": {
                "strengths": "Present moment awareness, sensory engagement, quick adaptation to environment",
                "challenges": "Long-term vision, recognizing abstract patterns, strategic planning"
            }
        }
        info = strengths_challenges.get(pairing, {"strengths": "Unknown", "challenges": "Unknown"})
        return f"Strengths: {info['strengths']}\nChallenges: {info['challenges']}"



   

# Usage example
def main():
    profiler = PsychologicalProfiler()

    conversation = [
        "I always try to analyze things logically, but I struggle in social situations.",
        "I'm excited about the future possibilities, but I have trouble remembering specific details from the past.",
        "I feel most comfortable when I have a clear plan and structure for my day.",
        "I often find myself daydreaming about potential innovations and new ideas.",
        "In team settings, I prefer to focus on the task at hand rather than engaging in small talk."
    ]

    for i, message in enumerate(conversation):
        try:
            context = "work" if i % 2 == 0 else "personal"
            profile, certainty = profiler.update_profile(message, context)
            print(f"New message: {message}")
            print(f"Context: {context}")
            print(f"Updated Profile:\n{profile}")
            print(f"Certainty: {certainty:.2f}")
            print("---")
        except Exception as e:
            print(f"Error processing message: {e}")
            print("---")

if __name__ == "__main__":
    main()