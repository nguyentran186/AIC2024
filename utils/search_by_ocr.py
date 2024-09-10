import json
import os
from rapidfuzz import process, fuzz

def search_by_ocr(search_query, ocr_data, limit=100):
    result = process.extract(search_query, ocr_data.values(), limit=limit, scorer=fuzz.partial_ratio)
    
    results = []
    indices = []
    for match, score, index in result:
        key = list(ocr_data.keys())[index]
        results.append(key)
        indices.append(index)
    return results, indices