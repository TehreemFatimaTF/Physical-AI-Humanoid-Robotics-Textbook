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

def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant_client.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]

# Test
print(retrieve("What data do you have?"))
