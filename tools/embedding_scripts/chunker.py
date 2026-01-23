from typing import List, Dict
from transformers import AutoTokenizer

def chunk_text_tokens(
    title,
    section,
    text,
    tokenizer,
    chunk_size=250,
    overlap=25,
    add_special_tokens=False
) -> List[Dict]:
    """
    Chunk text into overlapping token-based chunks.

    Args:
        text (str): Input text.
        tokenizer: Hugging Face tokenizer.
        chunk_size (int): Tokens per chunk.
        overlap (int): Overlapping tokens between chunks.
        add_special_tokens (bool): Whether to add special tokens ([CLS], [SEP]).

    Returns:
        list[str]: List of decoded text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Tokenize once
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None
    )["input_ids"]

    chunks = []
    chunk_id = 1
    step = chunk_size - overlap

    for i in range(0, len(token_ids), step):

        chunk_ids = token_ids[i:i + chunk_size]

        if len(chunk_ids) < overlap:
            break

        # Optionally add special tokens per chunk
        if add_special_tokens:
            chunk_tokens = tokenizer.build_inputs_with_special_tokens(chunk_ids)

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        chunks.append({
            'chunk_id': chunk_id,
            'input_ids': chunk_ids,
            'attention_mask': [1] * len(chunk_ids),
            'title': title,
            'section': section,
            'genes': ['LDLR'],
            'text': chunk_text
        })

        chunk_id += 1

    return chunks