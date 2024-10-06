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
            Analyze the following conversation and provide a brief assessment of the possible cognitive functions (Ti, Te, Fi, Fe, Ni, Ne, Si, Se) being used. 
            
            Conversation: {conversation}

            For each function that you believe might be present, provide:
            1. The function name
            2. A brief explanation of how it might be manifesting in the conversation (1-2 sentences)

            You don't need to mention all functions if you don't see evidence for them. Focus on the functions that seem most relevant to the conversation.

            Present your analysis in a clear, concise format.
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
            PRIMARY_PAIRING: [Hero function] Hero - [Inferior function] Inferior
            PAIRING_EXPLANATION: [Brief explanation of why this pairing was chosen, including key evidence from the analysis]

            Ensure both sections are filled out. The explanation should be concise but informative.
            """
        )

        # Initialize other components
        self.pairing_counts = Counter()
        self.total_assessments = 0
        self.conversation_data = []


    def identify_functions_with_reasoning(self, conversation: str) -> str:
        prompt = self.function_identification_prompt.format(conversation=conversation)
        return self.llm.invoke(prompt)
        
    def analyze_pairings_with_reasoning(self, function_analysis: str, conversation: str) -> str:
        return self.llm.invoke(self.pairing_analysis_prompt.format(
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
            "primary_pairing": "",
            "pairing_explanation": ""
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("PRIMARY_PAIRING:"):
                result["primary_pairing"] = line.split(':', 1)[1].strip()
            elif line.startswith("PAIRING_EXPLANATION:"):
                result["pairing_explanation"] = line.split(':', 1)[1].strip()
        
        return result
        
    def _validate_json_structure(self, parsed_json: Dict):
        """Validate the structure of the parsed JSON."""
        assert 'primary_pairing' in parsed_json, "Missing 'primary_pairing'"
        assert 'pairing_explanation' in parsed_json, "Missing 'pairing_explanation'"
        assert parsed_json['primary_pairing'], "Primary pairing is empty"
        assert parsed_json['pairing_explanation'], "Pairing explanation is empty"

    def _get_default_json_structure(self) -> Dict:
        """Return a default JSON structure."""
        return {
            "primary_pairing": "Unknown Hero - Unknown Inferior",
            "pairing_explanation": "No data available"
        }


    def update_profile(self, new_message: str, context: str = "general") -> Dict:
        """
        Update the psychological profile with a new message.

        Parameters:
        - new_message (str): The new conversation message to analyze.
        - context (str): The context of the message (e.g., work, personal).

        Returns:
        - Dict: The updated profile as a JSON object, including the certainty score.
        """
        try:
            print("New message: ", new_message)
            # Step 1: Function Identification with Reasoning
            function_analysis = self.identify_functions_with_reasoning(new_message)
            print("Function Analysis: ", function_analysis)
            
            # Step 2: Pairing Analysis with Reasoning
            pairing_analysis = self.analyze_pairings_with_reasoning(function_analysis, new_message)
            print("Pairing Analysis: ", pairing_analysis)
            
            # Step 3: Generate Structured Output
            structured_output = self.generate_structured_output(pairing_analysis)
            print("Structured Output: ", structured_output)
            
            # Extract the most likely pairing
            most_likely_pairing = structured_output.get('primary_pairing', 'Unknown - Unknown')
            print("Most Likely Pairing: ", most_likely_pairing)
            
            # Update counts
            self.pairing_counts[most_likely_pairing] += 1
            self.total_assessments += 1
            
            # Store conversation data
            self.conversation_data.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'message': new_message,
                'context': context,
                'most_likely_pairing': most_likely_pairing,
                'analysis': structured_output
            })
            
            # Generate profile and calculate certainty
            profile = self.generate_profile()
            certainty = self.calculate_certainty()
            
            # Add certainty to the profile
            profile["certainty"] = round(certainty, 2)
            
            return profile
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
        
        # Max entropy is log2(number of possible pairings)
        max_entropy = math.log2(len(PAIRINGS))
        
        # Normalize entropy to [0, 1] range and invert
        normalized_entropy = 1 - (entropy / max_entropy)
        
        # Factor in the amount of data
        data_factor = 1 - math.exp(-self.total_assessments / 10)  # Asymptotically approaches 1
        
        certainty = normalized_entropy * data_factor
        
        return certainty
            
    

    def generate_profile(self) -> Dict:
        """
        Generate a JSON object representing the current psychological profile.

        Returns:
        - Dict: A JSON object containing the profile information.
        """
        if not self.total_assessments:
            return {
                "most_likely_pairing": "Unknown Hero - Unknown Inferior",
                "probability": 0.0,
                "total_assessments": 0,
                "strengths": "No data available",
                "challenges": "No data available",
                "certainty": 0.0  # Initialize with 0.0
            }
        
        most_common_pairing, count = self.pairing_counts.most_common(1)[0]
        probability = count / self.total_assessments
        strengths, challenges = self.get_strengths_and_challenges(most_common_pairing).split('\n')
        strengths = strengths.replace("Strengths: ", "")
        challenges = challenges.replace("Challenges: ", "")
        
        profile = {
            "most_likely_pairing": most_common_pairing,
            "probability": round(probability, 2),
            "total_assessments": self.total_assessments,
            "strengths": strengths,
            "challenges": challenges
            # 'certainty' will be added in update_profile
        }
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
                "strengths": "Confident in their ability to filter out falsehood, examine axioms and frameworks, discover and resolve contradictions and align oneself with truth, Logical analysis, problem-solving, or objective decision-making",
                "challenges": "Insecurity in their ability to maintain and strengthen social coherence for communal and mutual success, Emotional expression, social harmony, or empathy in relationships"
            },
            "Fe Hero - Ti Inferior": {
                "strengths": "Confident in their ability to maintain and strengthen social coherence for communal and mutual success, Social awareness, empathy, or creating harmony in groups",
                "challenges": "Insecure in their ability to filter out falsehood, examine axioms and frameworks, discover and resolve contradictions and align oneself with truth, Logical analysis, maintaining objectivity, or individual decision-making"
            },
            "Te Hero - Fi Inferior": {
                "strengths": "Confident in their ability for to get objective functional success within the defined parameters of a specified and desired outcome, Efficient organization, strategic planning, or objective goal-setting",
                "challenges": "Insecure in their ability to discover, prioritise, reprioritize, recognize and process personal feelings, priorities and values; to evaluate, be accountable and align oneself with them, Personal value alignment, emotional self-awareness, or moral decision-making"
            },
            "Fi Hero - Te Inferior": {
                "strengths": "Confident in their ability to discover, prioritise, reprioritize, recognize and process personal feelings, priorities and values; to evaluate, be accountable and align oneself with them, has Strong personal values, authenticity, moral integrity",
                "challenges": "Insecure in their ability for to get objective functional success within the defined parameters of a specified and desired outcome, External organization, efficiency in systems, or objective goal-setting"
            },
            "Ne Hero - Si Inferior": {
                "strengths": "Confident in their ability to perceive the multitude and variance of perspectives, uncertainties and possibilities, how they interrelate and what they imply, Idea generation, seeing possibilities, or forming abstract connections",
                "challenges": "Insecure in their ability to discover, assign, create and associate meaning and consistency to the physical world, its impressions and effects and to create and maintain reliable and comfortable systems and habits for daily physical life, Detailed memory recall, maintaining routines, or practical application of past experiences"
            },
            "Si Hero - Ne Inferior": {
                "strengths": "Confident in their ability to discover, assign, create and associate meaning and consistency to the physical world, its impressions and effects and to create and maintain reliable and comfortable systems and habits for daily physical life, Detailed memory, establishing routines, or the practical application of past experiences",
                "challenges": "Insecure in their ability to perceive the multitude and variance of perspectives, uncertainties and possibilities, how they interrelate and what they imply, Generating new ideas, seeing abstract possibilities, adapting to change"
            },
            "Ni Hero - Se Inferior": {
                "strengths": "Confident in their ability to condense conceptual information to its most ideal and essential perspective, Having Long-term vision, pattern recognition, strategic foresight",
                "challenges": "Insecure in their ability to have clarity and lucidity on objective physical reality and present circumstances, see them for what they are and know how to affect them in return, Present moment awareness, sensory engagement, adapting to immediate environment"
            },
            "Se Hero - Ni Inferior": {
                "strengths": "Confident in their ability to have clarity and lucidity on objective physical reality and present circumstances, see them for what they are and know how to affect them in return. Present moment awareness, sensory engagement, quick adaptation to environment",
                "challenges": "Insecure in their ability to condense conceptual information to its most ideal and essential perspective, Long-term vision, recognizing abstract patterns, strategic planning"
            }
        }
        info = strengths_challenges.get(pairing, {"strengths": "Unknown", "challenges": "Unknown"})
        return f"Strengths: {info['strengths']}\nChallenges: {info['challenges']}"
    


# Usage example
def main():
    profiler = PsychologicalProfiler()

    # One user's conversations
    conversation = [
        "transcript 1: I always try to analyze things logically, but I struggle in social situations.",
        "transcript 2: I'm excited about the future possibilities, but I have trouble remembering specific details from the past.",
        "I feel most comfortable when I have a clear plan and structure for my day.",
        "I often find myself daydreaming about potential innovations and new ideas.",
        "In team settings, I prefer to focus on the task at hand rather than engaging in small talk."
    ]

    for i, message in enumerate(conversation):
        try:
            context = "work" if i % 2 == 0 else "personal"  # Context category
            profile = profiler.update_profile(message, context)  # Now returns a single JSON object
            
            # Serialize profile to JSON string for display or storage
            profile_json_str = json.dumps(profile, indent=4)
            
            print(f"New message: {message}")
            print(f"Context: {context}")
            print(f"Updated Profile:\n{profile_json_str}")  # Save and update this profile for each user
            print("---")
        except Exception as e:
            print(f"Error processing message: {e}")
            print("---")

if __name__ == "__main__":
    main()