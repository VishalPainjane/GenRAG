from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

batch_size = 32

if device == "cuda":
    batch_size = 128

print(f"Loading the model on device: {device}")
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device=device)

embedding_model.to(device)

print("Model loaded successfully")

def embed_text(pages_and_chunks_over_min_token_len):
    global batch_size

    print("Embedding text chunks in batches")
    print(f"Batch size: {batch_size}")
    all_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Processing batches"):
        batch = all_chunks[i:i + batch_size]
        
        embeddings = embedding_model.encode(
            batch, 
            device=device, 
            convert_to_tensor=True,  
            show_progress_bar=False 
        )
        
        for j, emb in enumerate(embeddings):
            pages_and_chunks_over_min_token_len[i + j]["embedding"] = emb

    print("Embedding complete")
    return pages_and_chunks_over_min_token_len, None
