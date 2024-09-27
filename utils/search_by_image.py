# Import module
import os
import open_clip
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss

def search_by_image(image, model, preprocess, index, embeddings, top_k=100, IDselector=None):
    """Search for tags similar to the given query tags."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy().astype(np.float32)
    
    
    ###### SEARCHING
    if IDselector is None:
        score, indices = index.search(image_features, top_k)
    else:
        id_selector = faiss.IDSelectorArray(IDselector)
        score, indices = index.search(image_features, top_k, params=faiss.SearchParametersIVF(sel=id_selector))
    
    results = []    
    for i in indices[0]:
        key = list(embeddings.keys())[i]
        results.append(key)
    
    return results, indices[0]