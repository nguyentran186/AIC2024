import copy
import time
import json
import requests
import numpy as np
import faiss
import open_clip
import glob
import torch
from PIL import Image
import os
import csv
import cv2
import sys

import gensim.downloader as api

from flask_cors import CORS
from flask import Flask, request, jsonify
from utils.search_by_tag import search_tags, embed_tags
from utils.search_by_text import search_by_text, search_by_sequence
from utils.search_by_ocr import search_by_ocr
from utils.search_by_image import search_by_image
from utils.search_by_sketch_text import search_by_sketch_text


sys.path.append('./tsbir/code')
from clip.model import convert_weights, CLIP
from clip.clip import _transform, load

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Global variables to store the models and indexes
visual_encoder = None
text_encoder = None
frames_index = None
frame_task_index = None
tag_index = None
visual_embeddings = None
tag_embeddings = None
first_request = True
ocr_data = None
sketch_text_encoder = None
task_preprocess = None



def load_sketch_text_encoder():
    model_config_file = './tsbir/code/training/model_configs/ViT-B-16.json'
    model_file = './tsbir/model/tsbir_model_final.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"
    print(f"Check device load sketch and text: {device}")

    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
            
    model = CLIP(**model_info)

    checkpoint = torch.load(model_file, map_location=device)

    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)

    model.eval()

    model = model.to(device)
    convert_weights(model)
    preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    return model, preprocess_val

def load_resources():
    print("Loading resource")
    global visual_encoder, text_encoder, frames_index, tag_index, visual_embeddings, tag_embeddings, ocr_data, visual_preprocess, sketch_text_encoder, frame_task_index, task_preprocess

    frame_index_path = "dict/faiss_trans_b3.bin"
    frame_task_path = "dict/faiss_TASK.bin"

    dataset_base_dir = 'AI-Challenge-fe/public/data/keyframes_trans/'
    npy_base_dir = 'dict/CLIP_features/CLIP_features'
    keyframes_dir = 'dict/tags/'

    #### Load Model #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print("Check device load resource: ", device)
    visual_encoder, _, visual_preprocess = open_clip.create_model_and_transforms('ViT-L-14', device=device, pretrained='datacomp_xl_s13b_b90k')
    # text_encoder = api.load("word2vec-google-news-300")

    frames_index = faiss.read_index(frame_index_path)
    frame_task_index = faiss.read_index(frame_task_path)
    # tag_index = faiss.read_index(tag_index_path)
    # Key frames embedding
    # Initialize the mapping dictionary
    frame_to_feature_map = {}

    # Loop over each L0x directory
    for level_dir in sorted(os.listdir(dataset_base_dir)):
        level_id = level_dir
        # Loop over each video directory within the L0x directory
        for video_dir in sorted(os.listdir(os.path.join(dataset_base_dir, level_dir))):
            video_id = video_dir

            # Load the features from the corresponding npy file
            npy_file_path = os.path.join(npy_base_dir, level_id, f'{video_id}.npy')
            if os.path.exists(npy_file_path):
                # features = np.load(npy_file_path)
                features = ""
            else:
                print(f"Warning: Missing npy file for {level_id}_{video_id}")
                continue

            # Get the list of frames in the video directory
            frame_files = sorted(os.listdir(os.path.join(dataset_base_dir, level_dir, video_dir)))

            # Map each frame to its corresponding feature
            for i, frame_file in enumerate(frame_files):
                if frame_file.endswith('.jpg'):
                    frame_index = os.path.splitext(frame_file)[0]
                    key = f"{level_id}_{video_id}_{frame_index}"
                    # frame_to_feature_map[key] = features[i].reshape(1,-1)
                    frame_to_feature_map[key] = ""
    visual_embeddings = frame_to_feature_map  
    print("TASK model and preprocess")
    sketch_text_encoder, task_preprocess = load_sketch_text_encoder() 
    print("Loaded resource")

initialized = False
@app.before_request
def initialize():
    global initialized
    if not initialized:
        initialized = True
        load_resources()
@app.route('/export', methods=['POST'], strict_slashes=False)
def export_csv():
    data = request.get_json()
    img_list = data.get('images')
    qa_data = data.get('textData')
    qno_data = data.get('questionData')
    
    with open('dict/keyframes_official.json', 'r') as f:
        key_frame_mapping = json.load(f)
        
    real_video_mapping = []
    
    if len(qa_data) == 0:
        for video_name in img_list:
            split = video_name.split('_')
            video_real_name = f'{split[0]}_{split[1]}'
            frame_index = str(int(split[2]))
            # Get the mapping for the video if it exists
            video_mapping = key_frame_mapping.get(video_real_name)
            if video_mapping:
                frame = video_mapping.get(frame_index)
                # Append the video name and its mapping to the real_video_mapping list
                real_video_mapping.append((video_real_name,frame))
    else:
        for video_name in img_list:
            split = video_name.split('_')
            video_real_name = f'{split[0]}_{split[1]}'
            frame_index = str(int(split[2]))
            # Get the mapping for the video if it exists
            video_mapping = key_frame_mapping.get(video_real_name)
            if video_mapping:
                frame = video_mapping.get(frame_index)
                # Append the video name and its mapping to the real_video_mapping list
                real_video_mapping.append((video_real_name,frame,qa_data))
    
    output_dir = 'submission'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if len(qno_data) != 0:
        csv_name = f'query-{qno_data}-kis.csv' if len(qa_data) == 0 else f'query-{qno_data}-qa.csv'
    else:
        csv_name = "untiled.csv"
    csv_file_path = os.path.join(output_dir, csv_name)
    # Write to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the rows
        for row in real_video_mapping:
            writer.writerow(row)
    
    return "CSV Exported Successfully"
    
@app.route('/search', methods=['POST'], strict_slashes=False)
def text_search():
    print("text search")
    data = request.get_json()
    # data = request.form
    print(data)    
    tag_search = data.get('tag_search') ## True/False
    prompt_search = data.get('prompt_search') ## True/False
    ocr_search = data.get('ocr_search')
    
    ocr_k = int(data['ocr_k'])
    tag_k = int(data['tag_k']) # Number
    prompt_k = int(data['prompt_k']) # Number
    
    ocr_query = str(data['ocr_query'])
    tag_query = str(data['tag_query']) # Str
    prompt_query = str(data['prompt_query']) # Str
    translate = data.get('translate')
    
    
    if ocr_search:
        print("Searching by ocr")
        ocr_results, ocr_indices = search_by_ocr(ocr_query, ocr_data, ocr_k)
        if prompt_search:
            print("Search by prompt")
            visual_indices = []
            for key in ocr_results:
                try:
                    # Attempt to find the index of the key
                    visual_index = list(visual_embeddings.keys()).index(key)
                    visual_indices.append(visual_index)
                except ValueError:
                    # Handle the case where the key is not found
                    print(f"Warning: Key '{key}' not found in visual_embeddings")
                    # Continue to the next key without appending
                    continue
            prompt_results, prompt_indices = search_by_text(prompt_query, visual_encoder, frames_index, visual_embeddings, prompt_k, translate, visual_indices)
            return jsonify(prompt_results)
        else:
            return jsonify(ocr_results)
    elif tag_search:
        print("Searching by tag")
        tag_results, tag_indices = search_tags(tag_query, text_encoder, tag_index, tag_embeddings, tag_k)
        if prompt_search:
            print("Searching by prompt")
            visual_indices = []
            for key in tag_results:
                try:
                    # Attempt to find the index of the key
                    visual_index = list(visual_embeddings.keys()).index(key)
                    visual_indices.append(visual_index)
                except ValueError:
                    # Handle the case where the key is not found
                    print(f"Warning: Key '{key}' not found in visual_embeddings")
                    # Continue to the next key without appending
                    continue
            prompt_results, prompt_indices = search_by_text(prompt_query, visual_encoder, frames_index, visual_embeddings, prompt_k, translate, visual_indices)
            return jsonify(prompt_results)
        else:
            return jsonify(tag_results)
    else:
        print("Searching by prompt")
        prompt_results, prompt_indices = search_by_text(prompt_query, visual_encoder, frames_index, visual_embeddings, prompt_k, translate)
        return jsonify(prompt_results)
    
@app.route('/search_by_image', methods=['POST'], strict_slashes=False)
def image_search():
    print("image search")

    # Check if 'image' is in the request.files
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image_file = request.files['image']  # Get the uploaded image

    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    image_bytes = image_file.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    image_results, image_indices = search_by_image(image, visual_encoder, visual_preprocess, frames_index, visual_embeddings)
    return jsonify(image_results)
    
@app.route('/search_by_sequence', methods=['POST'], strict_slashes=False)    
def sequence_search():
    data = request.get_json()
    image_results = search_by_sequence(data['queries'], visual_encoder, frames_index, visual_embeddings, False)
    return jsonify(image_results)

@app.route('/search_by_sketch_text', methods=['POST'], strict_slashes=False)
def sketch_text_search():
    data = request.get_json()
    image_results = search_by_sketch_text(data, sketch_text_encoder, task_preprocess, frame_task_index, visual_embeddings)
    return jsonify(image_results[0])


# Running app
if __name__ == '__main__':
    app.run(debug=True, port=8080)
