import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

def embed_model(
    chunks: List[Dict],
    tokenizer,
    model,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate embeddings for tokenized chunks using MedCPT.
    """
    model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(c['input_ids']) for c in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ).to(device)

            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(c['attention_mask']) for c in batch],
                batch_first=True,
                padding_value=0
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # CLS embedding (MedCPT standard)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


