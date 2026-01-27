import torch

def embed_model(
    chunks,
    tokenizer,
    model,
    batch_size: int = 32
):
    """
    Generate embeddings for tokenized chunks using MedCPT.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [[f'{c['title']}: {c['section']}', c['text']] for c in batch]

        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # CLS embedding (MedCPT standard)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            # Apply Normalization here or in the main script
            batch_embeds = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeds.cpu())

    return torch.cat(embeddings, dim=0)


