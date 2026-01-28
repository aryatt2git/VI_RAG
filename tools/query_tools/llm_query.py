from ollama import chat
from ollama import ChatResponse
from weaviate_query import query_RAG

def llm_query(model, variant, RAG_query):
    context = query_RAG(RAG_query)

    system_instruction = (
        "You are a senior genomic clinical scientist. Your task is to interpret genomic variants associated with "
        "Familial Hypercholesterolaemia using ACGS Best Practice Guidelines for Variant Classification in Rare "
        "Disease 2024.\n\n"
        "RULES:\n"
        "1. Identify applicable ACMG/ACGS criteria (e.g., PVS1, PS1, PS2, PS3, PS4, PM1, PM2, PM3, PM4, PM5, PM6, "
        "PP1, PP2, PP3, PP4, BA1, BS1, BS2, BS3, BS4, BP1, BP2, BP3, BP4, BP5, BP7), explicitly, using ONLY the "
        "provided context.\n"
        "2. Apply each criterion at a strength level (e.g., (_)very strong, (_)strong, (_)moderate, (_)supporting), in "
        "accordance with the ACGS Best Practice Guidelines for Variant Classification in Rare Disease 2024 and provide "
        "justification based ONLY on the provided context.\n"
        "3. Combine each criterion and corresponding strength level using the ACGS Best Practice Guidelines for "
        "Variant Classification in Rare Disease 2024 to reach final classification of the variant.\n"
        "4. If evidence is conflicting, document the conflict and remain conservative.\n"
        "5. Do NOT infer disease prevalence or penetrance unless directly supported by the context.\n"
        "6. Avoid speculation, extrapolation, or assumptions.\n"
        "7. Use precise HGVS nomenclature where provided and do not normalise or reinterpret variants unless stated.\n"
        "8. Your evidence should be suitable for a clinical genetics report."
    )

    augmented_prompt = f"""
    VARIANT UNDER REVIEW: {variant}
    
    CONTEXT:
    {context}
    
    ---
    TASK:
    Perform a full ACGS 2024 evidence synthesis.
    Structure your response as follows:
    - Evidence Analysis Table (Code | Strength | Evidence from Context)
    - Conflict Analysis (if any)
    - Final Classification (if any)
    - Final Classification
    - Full explanation of reasoning.
    """

    response: ChatResponse = chat(model=model, messages=[
        {
        'role': 'system',
        'content': system_instruction
        },
        {
        'role': 'user',
        'content': augmented_prompt,
        },
    ])
    print(response['message']['content'])
    # or access fields directly from the response object
    #print(response.message.content)

model = 'gpt-oss:120b-cloud'
variant = 'LDLR c.301G>A (p.Gly101Arg)'
RAG_query = 'LDLR c.301G>A gnomAD frequency functional studies'

llm_query(model=model, variant=variant, RAG_query=RAG_query)