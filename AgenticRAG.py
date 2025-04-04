import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain.embeddings.base import Embeddings

class CustomHFEmbedding(Embeddings):
    def embed_query(self, text: str) -> list:
        return get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(t) for t in texts]


# ------------------ Pinecone Setup ------------------ #
import pinecone
from pinecone import Pinecone, ServerlessSpec

index_name = "medigraphai"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# ------------------ Embedding Function ------------------ #
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled.squeeze().tolist()

# ------------------ Vector Store Wrapper ------------------ #
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore(index=index, embedding=CustomHFEmbedding())

# ------------------ Tools Setup ------------------ #
from langchain.agents import Tool

# Tool 1: Pinecone Retriever
def pinecone_retriever(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant documents found in Pinecone."
    combined_text = ""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        combined_text += f"[Result {i} from Pinecone]\nText: {doc.page_content}\nSource: {source}\n\n"
    return combined_text

pinecone_tool = Tool(
    name="PineconeRetriever",
    func=pinecone_retriever,
    description="Retrieves relevant healthcare documents from the Pinecone vector database."
)

# Tool 2: Tavily Web Search
from langchain_community.tools.tavily_search import TavilySearchResults

def tavily_search(query: str) -> str:
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
    results = search_tool.run(query)
    return f"[Web Search Results]\n{results}"

tavily_tool = Tool(
    name="WebSearch",
    func=tavily_search,
    description="Search web if Pinecone fails, Performs a web search for healthcare information using Tavily API and scrapes the top websites."
)

# ------------------ Gemini Flash LLM Setup ------------------ #
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)

# ------------------ ReAct Agent Prompt ------------------ #
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template("""
You are a helpful medical assistant. You have access to the following tools:
{tools}

Use the following format:

Question: {input}
Thought: your reasoning
Action: the tool to use, one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
If one tool fails to return useful information (e.g., returns a message like "No relevant documents found"), try the next most appropriate tool.

Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

{agent_scratchpad}
""")


# ------------------ Create Agent ------------------ #
from langchain.agents import AgentExecutor, create_react_agent

tools = [pinecone_tool, tavily_tool]

agent = create_react_agent(llm=llm, tools=tools, prompt=custom_prompt,)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ------------------ RAG Execution ------------------ #
def run_agentic_rag(query: str):
    response = agent_executor.invoke({"input": query})
    return response

# ------------------ Main ------------------ #
def main():
    user_query = "What are some natural remedies for acne?"
    answer = run_agentic_rag(user_query)
    print("\nFinal Answer:\n")
    print(answer["output"])

if __name__ == "__main__":
    main()
