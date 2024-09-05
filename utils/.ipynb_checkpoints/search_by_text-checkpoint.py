# Import module
import os
import open_clip
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss

def search_by_text(query_tags, model, index, embeddings, top_k=5, IDselector=None):
    """Search for tags similar to the given query tags."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_tokens = open_clip.tokenize(query_tags).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().detach().numpy().astype(np.float32)
    
    
    ###### SEARCHING
    if IDselector is None:
        score, indices = index.search(text_features, top_k)
    else:
        id_selector = faiss.IDSelectorArray(IDselector)
        score, indices = index.search(text_features, top_k, params=faiss.SearchParametersIVF(sel=id_selector))
    
    results = []
    for i in indices[0]:
        key = list(embeddings.keys())[i]
        results.append(key)
    
    return results, indices[0]