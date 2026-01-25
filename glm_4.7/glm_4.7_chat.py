from ollama import chat
from ollama import ChatResponse

def LLM_query(model, RAG_query, LLM_query)
    user_query = LLM_query
    augmented_prompt = f"""
    Use the following context to answer the question. 
    ---
    CONTEXT:
    {context}
    ---
    QUESTION: {user_query}
    
    Remember: ONLY use the context above. If not found, say you don't know.
    """

    response: ChatResponse = chat(model='glm-4.7:cloud', messages=[
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

LLM_query = 'pretend you are a genomic clinical scientist. interpret the variant c.301G>A in LDLR, with regard to its association with familial hypercholesterolaemia, using ACGS 2024 variant interpretation guidelines.'
RAG_query = 'c.301G>A in LDLR'