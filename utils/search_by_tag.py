import faiss
import gensim.downloader as api
import os
import json
import numpy as np
import h5py

def get_tag_vector(tag, model):
    """Get the vector for a single tag."""
    if tag in model:
        return model[tag]
    else:
        return np.zeros(model.vector_size)  # Return zero vector if tag not in model

def embed_tags(tags, model):
    """Embed a set of tags into a single vector by averaging."""
    vectors = [get_tag_vector(tag, model) for tag in tags]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # Return zero vector if no tags
    
def search_tags(query_tags, model, index, embeddings, top_k=1000):
    """Search for tags similar to the given query tags."""
    words = query_tags.split()
    query_vector = embed_tags(words, model)
    query_vector = np.expand_dims(query_vector, axis=0).astype('float32')
    
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i in indices[0]:
        key = list(embeddings.keys())[i]
        results.append(key)
    
    return results, indices[0]