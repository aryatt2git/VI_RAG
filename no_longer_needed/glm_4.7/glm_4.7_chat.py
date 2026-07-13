from VI_RAG.no_longer_needed.weaviate_query import query_RAG
from ollama import chat
from ollama import ChatResponse

def LLM_query(model, RAG_query, LLM_query):
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
        'content': 'You are a restricted assistant. Only use provided context.'
        }
        {
        'role': 'user',
        'content': augmented_prompt,
        },
    ])
    print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)

model = 'glm-4.7:cloud'
RAG_query = 'c.301G>A in LDLR'
LLM_query = 'pretend you are a genomic clinical scientist. interpret the variant c.301G>A in LDLR, with regard to its association with familial hypercholesterolaemia, using ACGS 2024 variant interpretation guidelines.'

LLM_query(model=model, RAG_query=RAG_query, LLM_query=LLM_query)