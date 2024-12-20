from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict
import json
import random
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

class PersonalityConversationGenerator:
    def __init__(self):
        self.llm = OpenAI(
            temperature=0.8,  # Slightly increased for more natural variation
            max_tokens=150
        )
        
        self.personality_templates = {
            "Ti Hero - Fe Inferior": {
                "background": """
                    You're someone who:
                    - Naturally thinks through problems systematically
                    - Has a dry sense of humor
                    - Can get deeply absorbed in interesting topics
                    - Values precision but isn't robotic
                    - Sometimes struggles with emotional expression but still cares deeply
                    - Has specific interests you're passionate about
                """,
                "conversation_style": "casual but thoughtful, occasionally sharing specific examples from your interests"
            },
            "Ne Hero - Si Inferior": {
                "background": """
                    You're someone who:
                    - Gets excited about possibilities but isn't always over-the-top
                    - Has varied interests that naturally come up in conversation
                    - Sometimes forgets practical details when excited about ideas
                    - Can be both playful and insightful
                    - Enjoys connecting different concepts
                    - Has some concrete experiences to draw from
                """,
                "conversation_style": "engaged and genuine, mixing enthusiasm with real observations"
            }
        }
        
        self.interviewer_template = """
        You're having a natural conversation with someone. Respond in a way that feels genuine and unscripted.
        Previous messages: {context}
        
        Keep your response conversational and natural, as if you're really talking to someone.
        Don't try to explicitly probe their personality - just have a real conversation.
        """
        
        self.target_template = """
        Background on your natural way of being:
        {background}
        
        You're having a casual conversation. The other person just said:
        "{message}"

        Previous context: {context}

        Respond naturally in your voice, bringing in specific examples or thoughts when relevant.
        Your style is {conversation_style}.
        The response should feel like something a real person would say in conversation.
        """
        
        self.interviewer_prompt = PromptTemplate(
            input_variables=["context"],
            template=self.interviewer_template
        )
        
        self.target_prompt = PromptTemplate(
            input_variables=["background", "message", "context", "conversation_style"],
            template=self.target_template
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_message(self, prompt_template: PromptTemplate, **kwargs) -> str:
        try:
            prompt = prompt_template.format(**kwargs)
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating message: {e}")
            raise

    def generate_conversation(
        self,
        personality_type: str,
        topics: List[str] = ["work", "hobbies", "current events", "technology", "food"],
        num_turns: int = 3
    ) -> List[Dict]:
        conversation = []
        personality = self.personality_templates[personality_type]
        current_topic = random.choice(topics)
        
        # Start with a natural conversation opener about the chosen topic
        initial_topics = {
            "work": "How's your week been going? Working on anything interesting?",
            "hobbies": "Did you get up to anything fun this weekend?",
            "current events": "Have you been following the news about the new space telescope?",
            "technology": "I just got a new phone and I'm still figuring it out. How do you like yours?",
            "food": "Have you tried any good restaurants lately?"
        }
        
        # Start the conversation
        conversation.append({
            "speaker": "interviewer",
            "message": initial_topics[current_topic]
        })
        
        for _ in range(num_turns - 1):  # -1 because we already added the opener
            try:
                # Generate target response
                context = "\n".join([msg["message"] for msg in conversation[-2:]])
                target_msg = self.generate_message(
                    self.target_prompt,
                    background=personality["background"],
                    message=conversation[-1]["message"],
                    context=context,
                    conversation_style=personality["conversation_style"]
                )
                
                if target_msg:
                    conversation.append({
                        "speaker": "target",
                        "message": target_msg,
                        "personality_type": personality_type
                    })
                
                time.sleep(1)
                
                # Generate interviewer response
                context = "\n".join([msg["message"] for msg in conversation[-2:]])
                interviewer_msg = self.generate_message(
                    self.interviewer_prompt,
                    context=context
                )
                
                if interviewer_msg:
                    conversation.append({
                        "speaker": "interviewer",
                        "message": interviewer_msg
                    })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in conversation generation: {e}")
                break
                
        return conversation

    def generate_dataset(
        self,
        num_conversations: int = 5,
        turns_per_conversation: int = 3,
        output_file: str = "conversations.json"
    ):
        """Generate multiple conversations"""
        all_conversations = []
        personality_types = list(self.personality_templates.keys())
        
        for i in range(num_conversations):
            personality_type = random.choice(personality_types)
            try:
                conversation = self.generate_conversation(
                    personality_type=personality_type,
                    num_turns=turns_per_conversation
                )
                
                if conversation:  # Only add if we got a valid conversation
                    all_conversations.append({
                        "id": i,
                        "personality_type": personality_type,
                        "messages": conversation
                    })
                
                # Add a small delay between conversations
                time.sleep(2)
                
            except Exception as e:
                print(f"Error generating conversation {i}: {e}")
                continue
        
        with open(output_file, 'w') as f:
            json.dump(all_conversations, f, indent=2)
            
        print(f"Generated {len(all_conversations)} conversations")

def main():
    generator = PersonalityConversationGenerator()
    generator.generate_dataset(
        num_conversations=3,
        turns_per_conversation=2,
        output_file="synthetic_conversations.json"
    )

if __name__ == "__main__":
    import time
    main()