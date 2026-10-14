from typing import List, Dict

def chunk_text_tokens(
    title,
    section,
    text,
    tokenizer,
    chunk_size=300,
    overlap=50
):

    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):

        chunk_tokens = tokens[i:i + chunk_size]
        text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "title": title,
            "section": f'{section}_{i}',
            "text": text,
        })

    return chunks