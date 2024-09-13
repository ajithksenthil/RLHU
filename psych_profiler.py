import os
import datetime
from typing import List, Dict, Tuple
import numpy as np
import json
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake

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
        self.llm = OpenAI(temperature=0.7)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            You are an expert in Jungian psychology. Analyze the following text for psychological insights related to Jungian cognitive functions:

            Text: {text}

            Provide your analysis in the following JSON format and **do not include any extra text**:
            {{
                "cognitive_functions": ["list", "of", "detected", "functions"],
                "confidence_areas": ["list", "of", "confidence", "areas"],
                "anxiety_areas": ["list", "of", "anxiety", "areas"],
                "problem_solving": "approach",
                "thinking_style": "style",
                "time_orientation": "orientation"
            }}
            
            Ensure all cognitive functions (Ti, Te, Fi, Fe, Ni, Ne, Si, Se) are considered.

            Example Output:
            {{
                "cognitive_functions": ["Ti", "Ne"],
                "confidence_areas": ["logical analysis", "idea generation"],
                "anxiety_areas": ["social interactions"],
                "problem_solving": "Analytical",
                "thinking_style": "Abstract",
                "time_orientation": "Future-focused"
            }}
            """
        )
        self.llm_chain = self.prompt | self.llm
        self.pairing_probabilities = np.ones(len(PAIRINGS)) / len(PAIRINGS)
        self.conversation_data = []
        self.current_profile = None
        self.current_certainty = 0
        self.bayesian_network = self.initialize_bayesian_network()
        self.time_constant = 3600  # 1 hour in seconds

    def initialize_bayesian_network(self):
        """
        Initialize the Bayesian network with appropriate nodes, edges, and CPDs.
        """
        model = BayesianNetwork()

        # Add nodes
        model.add_node('Pairing')
        for func in COGNITIVE_FUNCTIONS:
            model.add_node(func)
            model.add_edge('Pairing', func)

        # Define CPDs
        cpd_list = []

        # CPD for 'Pairing' (Uniform prior)
        cpd_pairing = TabularCPD(
            variable='Pairing',
            variable_card=8,
            values=[[1/8] for _ in range(8)]  # Correct shape (8, 1)
        )
        cpd_list.append(cpd_pairing)

        # Mapping from pairing index to functions
        pairing_to_functions = {
            0: {'Ti': 'Hero', 'Fe': 'Inferior'},
            1: {'Fe': 'Hero', 'Ti': 'Inferior'},
            2: {'Te': 'Hero', 'Fi': 'Inferior'},
            3: {'Fi': 'Hero', 'Te': 'Inferior'},
            4: {'Ne': 'Hero', 'Si': 'Inferior'},
            5: {'Si': 'Hero', 'Ne': 'Inferior'},
            6: {'Ni': 'Hero', 'Se': 'Inferior'},
            7: {'Se': 'Hero', 'Ni': 'Inferior'},
        }

        # Function to create CPD for a cognitive function
        def create_cpd_for_function(func_name):
            values = [[], []]  # P(func=0 | Pairing), P(func=1 | Pairing)
            for pairing_index in range(8):
                pairing_funcs = pairing_to_functions[pairing_index]
                if func_name in pairing_funcs:
                    role = pairing_funcs[func_name]
                    if role == 'Hero':
                        p_present = 0.9
                    elif role == 'Inferior':
                        p_present = 0.3
                else:
                    p_present = 0.5  # Neutral probability
                p_absent = 1 - p_present
                values[0].append(p_absent)  # P(func=0 | Pairing)
                values[1].append(p_present)  # P(func=1 | Pairing)
            cpd = TabularCPD(
                variable=func_name,
                variable_card=2,
                evidence=['Pairing'],
                evidence_card=[8],
                values=values
            )
            return cpd

        # Create and add CPDs for each cognitive function
        for func in COGNITIVE_FUNCTIONS:
            cpd_func = create_cpd_for_function(func)
            cpd_list.append(cpd_func)

        model.add_cpds(*cpd_list)
        model.check_model()  # Validate the model
        return model
    
    def update_profile(self, new_message: str, context: str = "general") -> Tuple[str, float]:
        """
        Updates the user's psychological profile based on a new message.

        Parameters:
        - new_message (str): The latest message from the user.
        - context (str): The context of the message (e.g., "work", "personal").

        Returns:
        - Tuple[str, float]: The updated profile and the current certainty level.
        """
        # Step 1: LLM Analysis
        llm_output = self.llm_chain.invoke({"text": new_message})
        
        # Step 2: Post-LLM Processing
        processed_data = self.process_llm_output(llm_output, new_message)
        
        # Step 3: Hero-Inferior Pairing Analysis
        self.update_pairing_probabilities(processed_data)
        
        # Step 4: Certainty Calculation
        self.current_certainty = self.calculate_certainty()
        
        # Step 5: Temporal Analysis
        self.temporal_analysis()
        
        # Step 6: Profile Generation
        self.current_profile = self.generate_profile()
        
        # Store conversation data
        self.conversation_data.append({
            'timestamp': datetime.datetime.now(),
            'message': new_message,
            'context': context,
            'processed_data': processed_data,
            'probabilities': self.pairing_probabilities.copy(),
            'certainty': self.current_certainty
        })
        
        return self.current_profile, self.current_certainty
    
    def process_llm_output(self, llm_output: str, original_text: str) -> Dict:
        """
        Process the LLM output and extract relevant information.

        Parameters:
        - llm_output (str): The raw output from the LLM.
        - original_text (str): The original input text.

        Returns:
        - Dict: Processed data including cognitive functions, sentiment, and topics.
        """
        # Remove any text before the first '{' to handle LLM extra text
        json_start = llm_output.find('{')
        if json_start != -1:
            llm_output = llm_output[json_start:]
        try:
            llm_data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            # Retry with a simplified prompt
            llm_data = self.retry_llm_with_simplified_prompt(original_text)
        
        processed_data = {
            'cognitive_functions': llm_data.get("cognitive_functions", []),
            'confidence_areas': llm_data.get("confidence_areas", []),
            'anxiety_areas': llm_data.get("anxiety_areas", []),
            'problem_solving': llm_data.get("problem_solving", "Unknown"),
            'thinking_style': llm_data.get("thinking_style", "Unknown"),
            'time_orientation': llm_data.get("time_orientation", "Unknown"),
            'sentiment': self.analyze_sentiment(original_text),
            'topics': self.extract_topics(original_text)
        }
        return processed_data

    def retry_llm_with_simplified_prompt(self, original_text: str) -> Dict:
        """
        Retry LLM analysis with a simplified prompt if the initial attempt fails.

        Parameters:
        - original_text (str): The original input text.

        Returns:
        - Dict: Simplified processed data.
        """
        simplified_prompt = PromptTemplate(
            input_variables=["text"],
            template="Analyze this text and list any cognitive functions mentioned: {text}"
        )
        simplified_chain = simplified_prompt | self.llm
        simplified_output = simplified_chain.invoke({"text": original_text})
        
        # Extract cognitive functions from the simplified output
        cognitive_functions = [func for func in COGNITIVE_FUNCTIONS if func in simplified_output]
        
        return {
            "cognitive_functions": cognitive_functions,
            "confidence_areas": [],
            "anxiety_areas": [],
            "problem_solving": "Unknown",
            "thinking_style": "Unknown",
            "time_orientation": "Unknown"
        }
    
    def update_pairing_probabilities(self, processed_data):
        """
        Update the probabilities of each Hero-Inferior pairing based on the processed data.

        Parameters:
        - processed_data (Dict): The processed data from LLM output.
        """
        evidence_funcs = processed_data['cognitive_functions']
        evidence = {}
        for func in evidence_funcs:
            evidence[func] = 1  # Cognitive function present

        # Do not set evidence for functions not mentioned
        # This way, unobserved functions remain as random variables

        infer = VariableElimination(self.bayesian_network)
        try:
            prob_query = infer.query(variables=['Pairing'], evidence=evidence)
            self.pairing_probabilities = prob_query.values
        except Exception as e:
            print(f"Error during inference: {e}")
            self.pairing_probabilities = np.ones(len(PAIRINGS)) / len(PAIRINGS)

    
    # Multiplicative certainty calculation
    def calculate_certainty(self) -> float:
        max_prob = np.max(self.pairing_probabilities)
        data_factor = 1 - np.exp(-len(self.conversation_data) / 10)
        consistency = self.calculate_consistency()
        # Ensure consistency is between 0 and 1
        consistency = max(0, consistency)
        certainty = max_prob * data_factor * consistency
        return min(max(certainty, 0.0), 0.99)
    
    
    """
    # Additive certainty calculation
    def calculate_certainty(self) -> float:
        max_prob = np.max(self.pairing_probabilities)
        data_factor = 1 - np.exp(-len(self.conversation_data) / 5)  # Adjusted decay rate
        consistency = self.calculate_consistency()
        consistency = max(0, consistency)  # Ensure consistency is non-negative
        certainty = (0.5 * max_prob) + (0.3 * data_factor) + (0.2 * consistency)
        return min(certainty, 0.99)

    """
    


    def temporal_analysis(self):
        """
        Perform temporal analysis on the conversation data, applying decay to older data points.
        """
        current_time = datetime.datetime.now()
        for data_point in self.conversation_data[:-1]:
            time_diff = (current_time - data_point['timestamp']).total_seconds()
            decay = np.exp(-time_diff / self.time_constant)
            data_point['probabilities'] *= decay

    def calculate_consistency(self) -> float:
        if len(self.conversation_data) < 2:
            return 1.0  # Maximum consistency when there's only one data point
        prev_probs = self.conversation_data[-2]['probabilities']
        curr_probs = self.pairing_probabilities
        # Calculate the cosine similarity instead of correlation
        numerator = np.dot(prev_probs, curr_probs)
        denominator = np.linalg.norm(prev_probs) * np.linalg.norm(curr_probs)
        similarity = numerator / denominator if denominator != 0 else 0
        return similarity
    

    """
    def calculate_consistency(self) -> float:
        if len(self.conversation_data) < 2:
            return 1.0
        prev_probs = self.conversation_data[-2]['probabilities']
        curr_probs = self.pairing_probabilities
        correlation = np.corrcoef(prev_probs, curr_probs)[0, 1]
        # Rescale correlation to be between 0 and 1
        rescaled_corr = (correlation + 1) / 2
        return rescaled_corr if not np.isnan(rescaled_corr) else 0.0
    """

    def generate_profile(self) -> str:
        """
        Generate a psychological profile based on the current probabilities and data.

        Returns:
        - str: A string representation of the psychological profile.
        """
        top_pairing_index = np.argmax(self.pairing_probabilities)
        top_pairing = PAIRINGS[top_pairing_index]
        profile = f"Most likely Hero-Inferior pairing: {top_pairing}\n"
        profile += f"Probability: {self.pairing_probabilities[top_pairing_index]:.2f}\n"
        profile += f"Certainty: {self.current_certainty:.2f}\n"
        profile += "Potential strengths and challenges:\n"
        profile += self.get_strengths_and_challenges(top_pairing)
        profile += "\nRecent topics of interest:\n"
        profile += ', '.join(self.get_recent_topics())
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

    def get_recent_topics(self) -> List[str]:
        """
        Get the most recent topics of interest from the conversation data.
    
        Returns:
        - List[str]: A list of recent topics.
        """
        recent_topics = []
        for data in self.conversation_data[-5:]:
            recent_topics.extend(data['processed_data']['topics'])
        # Remove duplicates and return up to 5 topics
        unique_topics = list(set(recent_topics))
        return unique_topics[:5]

    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of the given text.

        Parameters:
        - text (str): The text to analyze.

        Returns:
        - str: The sentiment category ('Positive', 'Negative', or 'Neutral').
        """
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound_score = scores['compound']

        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from the given text.

        Parameters:
        - text (str): The text to analyze.

        Returns:
        - List[str]: A list of extracted topics.
        """
        kw_extractor = yake.KeywordExtractor()
        keywords = kw_extractor.extract_keywords(text)
        # Extract top 5 keywords
        top_keywords = [kw for kw, score in sorted(keywords, key=lambda x: x[1])[:5]]
        return top_keywords

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
        context = "work" if i % 2 == 0 else "personal"
        profile, certainty = profiler.update_profile(message, context)
        print(f"New message: {message}")
        print(f"Context: {context}")
        print(f"Updated Profile:\n{profile}")
        print(f"Certainty: {certainty:.2f}")
        print("---")

if __name__ == "__main__":
    main()