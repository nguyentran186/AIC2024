import os
import numpy as np
import json
import torch
import sys
import base64
from PIL import Image
import io
from deep_translator import GoogleTranslator


sys.path.append('./tsbir/code')

##make sure CODE_PATH is pointing to the correct path containing clip.py before running 
from clip.model import convert_weights, CLIP
from clip.clip import tokenize


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

print(f"Check device: {device}")
def get_feature(model, query_sketch, query_text, transformer):
    
    print(type(transformer))
    print(type(query_sketch))
    print(type(transformer(query_sketch)))
    img1 = transformer(query_sketch).unsqueeze(0).to(device)

    txt = tokenize([str(query_text)])[0].unsqueeze(0).to(device)
    with torch.no_grad():
        sketch_feature = model.encode_sketch(img1)
        print("Finish encode sketch")
        text_feature = model.encode_text(txt)
        print("Finish encode text")
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
    print("Finish get feature")
    return model.feature_fuse(sketch_feature,text_feature)

def search_by_sketch_text(data, model, task_preprocess, indexes, embeddings):
    print(type(data['sketch']))
    #Decode base64 image and in new Image object
    data['sketch'] = base64.b64decode(data['sketch'])
    image = Image.open(io.BytesIO(data['sketch']))
    #replace transparent by white background
    # Create a white background image
    white_bg = Image.new("RGBA", image.size, "WHITE")

    # Paste the original image onto the white background
    white_bg.paste(image, (0, 0), image)

    # Convert back to RGB to remove transparency
    data['sketch'] = white_bg.convert('RGB')
    
    # Save for testing
    data['sketch'].save('test_with_white_background.png')
    print(data['sketch'].size)
    feature = get_feature(model, data['sketch'], data['text'], task_preprocess)
    
    score, indices = indexes.search(feature.cpu().detach().numpy().astype(np.float32), 100)
    results = []    
    for i in indices[0]:
        key = list(embeddings.keys())[i]
        results.append(key)
    
    return results, indices[0]