from ollama import chat
from ollama import ChatResponse
from weaviate_query import query_RAG

def llm_query(model, RAG_query, LLM_query):
    context = query_RAG(RAG_query)
    augmented_prompt = f"""
    Use the following context to answer the question. 
    ---
    CONTEXT:
    {context}
    ---
    QUESTION: {LLM_query}
    
    Remember: ONLY use the context above. If not found, say you don't know.
    """

    response: ChatResponse = chat(model=model, messages=[
        {
        'role': 'system',
        'content': 'You are a genomic clinical scientist. Only use provided context.'
        },
        {
        'role': 'user',
        'content': augmented_prompt,
        },
    ])
    print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)

model = 'gpt-oss:120b-cloud'
rag_query = 'c.301G>A in LDLR'
LLM_query = 'how many people have variants in LDLR that cause FH?'

llm_query(model=model, RAG_query=rag_query, LLM_query=LLM_query)