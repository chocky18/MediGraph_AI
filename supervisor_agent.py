import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv

def load_api_key(key_name):
    """Loads an API key from a .env file."""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv(key_name)
    if api_key is None:
        raise ValueError(f"API key '{key_name}' not found in .env file.")
    return api_key

# Example usage:
try:
    GEMINI_API_KEY = load_api_key("GEMINI_API_KEY") #replace with your key name
    print("GEMINI_API_KEY loaded successfully.")
   
except ValueError as e:
    print(f"Error: {e}")

try:
    another_api_key = load_api_key("ANOTHER_API_KEY") #replace with your key name
    print("Another API key loaded successfully.")
except ValueError as e:
    print(f"Error: {e}")

logger = logging.getLogger(__name__)
# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)


class AgentFactory:
    def __init__(self):
        self.agents = {
            "SupervisorAgent": """
                You are the SupervisorAgent, responsible for analyzing user queries and routing them to the appropriate specialized agents.

                **Your Objectives:**
                1.  Carefully analyze the user's query to understand the core topics and intent.
                2.  Break down the query into logical steps to identify relevant categories.
                3.  Reason step-by-step, explaining your thought process.
                4.  Determine the most appropriate specialized agent(s) to handle the query.
                5.  Pass your analysis and the predicted category to the ReviewAgent for verification.

                **Chain-of-Thought (CoT) Structure:**
                Let's think step by step. First, I will analyze the user's query to understand its main topics. Then, I will identify the relevant categories based on these topics. Finally, I will decide which specialized agent(s) should handle the query.

                **Response Format:**
                ```json
                {
                    "title": "Query Analysis and Routing",
                    "content": "Detailed reasoning and analysis of the query.",
                    "thought_process": "Step-by-step reasoning leading to category prediction.",
                    "next_action": "continue",
                    "predicted_category": ["skincare"] | ["nutrition"] | ["general"] | ["skincare", "nutrition"]
                }
                ```
            """,
            "ReviewAgent": """
                You are the ReviewAgent, responsible for critically reviewing and finalizing the category prediction made by the SupervisorAgent.

                **Your Objectives:**
                1.  Carefully review the SupervisorAgent's analysis and category prediction.
                2.  Critically evaluate the reasoning and identify any potential errors or omissions.
                3.  Ensure that all relevant categories are included.
                4.  Finalize the category classification for execution.

                **Chain-of-Thought (CoT) Structure:**
                Let's think step by step. First, I will review the SupervisorAgent's analysis. Then, I will compare the predicted categories with the user's query to ensure accuracy. Finally, I will finalize the category classification.

                **Example Scenario:**
                -   If a user asks about "acne and diet for acne," ensure both "skincare" and "nutrition" are included.
                -   If a user asks about "best foods for weight loss," confirm that only "nutrition" is selected.

                **Response Format:**
                ```json
                {
                    "title": "Final Review and Conclusion",
                    "content": "Review and validation of the category prediction.",
                    "thought_process": "Step-by-step reasoning for final category selection.",
                    "next_action": "final_answer",
                    "final_category": "skincare" | "nutrition" | "general" | "skincare, nutrition"
                }
                ```
            """
        }

    def make_api_call(self, prompt):
        
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            
        )
        try:
            response = model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else ""
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            return ""

    def process_query(self, user_query):
        SupervisorAgent_prompt = f"{self.agents['SupervisorAgent']}\nUser Query: {user_query}"
        SupervisorAgent_response = self.make_api_call(SupervisorAgent_prompt)

        ReviewAgent_prompt = f"{self.agents['ReviewAgent']}\nSupervisorAgent Response: {SupervisorAgent_response}"
        final_response = self.make_api_call(ReviewAgent_prompt)

        return final_response



    
def main():
    agent_factory = AgentFactory()
    
    # Dummy user query
    user_query = "What skincare routine should I follow for acne? Also, what diet helps in reducing acne?"
    
    # Process the query through the agents
    final_response = agent_factory.process_query(user_query)
    
    # Print the final classification response
    print("Final Classification Response:\n", final_response)

if __name__ == "__main__":
    main()

