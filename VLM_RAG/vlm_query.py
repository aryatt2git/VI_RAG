from ollama import chat
from ollama import ChatResponse
from weaviate_query import query_PDF_RAG

def vlm_query(model, RAG_query, LLM_query):
    context = query_PDF_RAG(RAG_query)

    print(context)

    # Safety check: Did we actually find anything?
    if not context:
        print("Error: No relevant clinical context found in the knowledge base.")
        return

    augmented_prompt = f"""
    Use the following context to answer the question. 
    ---
    CONTEXT:
    {context}
    ---
    QUESTION: {LLM_query}
    
    INSTRUCTIONS:
    - Answer the question as best you can using the attached images.
    """

    response: ChatResponse = chat(model=model, messages=[
        {
        'role': 'system',
        'content': 'You are a genomic clinical scientist. Use only provided evidence for clinical assertions.'
        },
        {
        'role': 'user',
        'content': augmented_prompt,
        'images': context
        },
    ])

    print("--- VLM RESPONSE ---")
    # print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)
    return response.message.content

"""
model = 'gpt-oss:120b-cloud'
rag_query = 'c.301G>A in LDLR'
LLM_query = 'pretend you are a genomic clinical scientist. interpret the variant c.301G>A in LDLR, with regard to its association with familial hypercholesterolaemia, using ACGS 2024 variant interpretation guidelines.'
"""

model = 'gpt-oss:120b-cloud'
rag_query = 'Is there any mention of familial hypercholesterolaemia?'
LLM_query = 'What is Familial Hypercholesterolaemia?'

vlm_query(model=model, RAG_query=rag_query, LLM_query=LLM_query)

