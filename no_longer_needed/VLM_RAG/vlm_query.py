import base64
from ollama import chat
from ollama import ChatResponse
from weaviate_query import query_PDF_RAG

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_query(model, RAG_query, LLM_query):
    context = query_PDF_RAG(RAG_query)

    #print(context)
    encoded_images = [encode_image(path) for path in context]

    # Safety check: Did we actually find anything?
    if not context:
        print("Error: No relevant clinical context found in the knowledge base.")
        return

    augmented_prompt = f"""
    Use the following context to answer the question. 
    ---
    CONTEXT:
    {encoded_images}
    ---
    QUESTION: {LLM_query}
    
    INSTRUCTIONS:
    - Answer the question as best you can using only the attached images.
    """

    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': 'You are a genomic clinical scientist. Use only provided evidence for clinical assertions.'
             },
            {
                'role': 'user',
                'content': augmented_prompt,
                'images': encoded_images
             }
        ],
        options={
            'temperature': 0.0,       # Lower temperature makes the output more focused and consistent
            'repeat_penalty': 1.0,    # Setting to 1.0 turns OFF the penalty that stops it from repeating itself
            'num_predict': 5000,      # Give the model plenty of room to write a long answer
            'seed': 42                # Uncomment this if you want the EXACT same answer every single time
        },
        keep_alive = 0                # <--- This forces a fresh start for the next run
    )

    print("--- VLM RESPONSE ---")
    # print(response['message']['content'])
    # or access fields directly from the response object
    print(response.message.content)
    return response.message.content


model = 'nemotron-3-super:cloud'
rag_query = ('Are c.301G>A Glu101Lys E101K Glu80Lys E80K mentioned in any papers, tables and figures? Please repeat '
             'this function two more times with different files.')
LLM_query = ("pretend you are a genomic clinical scientist. How many affected individuals are carriers of the c.301G>A "
             "variant in LDLR? Consider alternative descriptions of the variant that might appear in the literature "
             "including 'Glu101Lys', 'E101K', 'Glu80Lys' and 'E80K'.")

"""
model = 'gpt-oss:120b-cloud'
rag_query = 'give me a comprehensive description of familial hypercholesterolaemia'
LLM_query = 'give me a comprehensive description of familial hypercholesterolaemia'
"""

vlm_query(model=model, RAG_query=rag_query, LLM_query=LLM_query)

