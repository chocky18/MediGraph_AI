# from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ------------------ Gemini Flash LLM Setup ------------------ #
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)


async def main():
    agent = Agent(
        task="fetch the data of vitamin c serum from https://thedermatologystore.com/ and https://www.nykaa.com/",
        llm=llm,
    )
    await agent.run()

asyncio.run(main())