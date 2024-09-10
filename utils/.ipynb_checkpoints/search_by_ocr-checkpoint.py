import json
import os
from rapidfuzz import process, fuzz

def search_by_ocr(search_query, tag_embedding, limit=10):
    results = process.extract(search_query, tag_embedding.values(), limit=limit, scorer=fuzz.partial_ratio)
    return results