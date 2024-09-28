# Import module
import os
import open_clip
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss

from deep_translator import GoogleTranslator

def search_by_text(query_tags, model, index, embeddings, top_k=5, translate=False, IDselector=None):
    """Search for tags similar to the given query tags."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if translate:
        translator = GoogleTranslator(source='vi', target='en')
        query_tags = translator.translate(query_tags)
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


def search_by_sequence(query_sequence, visual_encoder, frames_index, visual_embeddings, translate):
    prompt_k = 100
    selected_frames = None
    for i, query in enumerate(query_sequence):
        candidate_frames, _ = search_by_text(query, visual_encoder, frames_index, visual_embeddings, top_k=prompt_k*(len(query_sequence)-i), translate=translate)
        if selected_frames is None:
            selected_frames = [(f, int(f.split('_')[2])) for f in candidate_frames]
        else:
            temp = []
            for selected_frame in selected_frames:
                selected_frame_level, selected_frame_video, _ = selected_frame[0].split('_')
                selected_frame_index = int(selected_frame[1])
                for frame in candidate_frames:
                    frame_level, frame_video, frame_index = frame.split('_')
                    frame_index = int(frame_index)
                    if frame_level == selected_frame_level and frame_video == selected_frame_video:
                        if frame_index > selected_frame_index and frame_index <= selected_frame_index+10:
                            temp.append((selected_frame[0], frame_index))
                            break
            
            selected_frames = temp
    print(f"Cmn: {selected_frames}")
    return [f[0] for f in selected_frames]
                        

    