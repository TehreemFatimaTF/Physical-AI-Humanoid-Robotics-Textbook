from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import asyncio



load_dotenv()
set_tracing_disabled(disabled=True)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
provider = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    openai_client=provider
)

import cohere
from qdrant_client import QdrantClient

# Initialize Cohere client
cohere_client = cohere.Client("xzwpz0gbkUMQl3Z9V1gfPQlOPDVk7WVLaggg0wbp")
# Connect to Qdrant
qdrant_client = QdrantClient(
    url="https://911355da-9528-4509-be9f-968bf18c9bcb.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.U5k0Jvt7IjxQ5zxM_sW_OX1hk0xS7llAnS8p6uzs4Q0",
)

def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


@function_tool
def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant_client.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]



agent = Agent(
    name="Assistant",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.
To answer the user question, first call the tool `retrieve` with the user query.
Use ONLY the returned content from `retrieve` to answer.
If the answer is not in the retrieved content, say "I don't know".
""",
    model=model,
    tools=[retrieve]
)
async def main():
    result = await Runner.run(
        agent,
        input="what is physical"
    )
    print(result.final_output)

# Run the async function
asyncio.run(main())

