

import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "prepared_blacklist.json"

# ---------- NATASHA INIT ----------

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

# ---------- MODEL ----------

model = SentenceTransformer(MODEL_NAME)

# ---------- PREPROCESSING ----------

def remove_numbering(text):
    return re.sub(r'^\d+[\.\)]?\s*', '', text)

def remove_after_solution(text):
    return re.sub(r'\bрешение\b.*', '', text, flags=re.IGNORECASE)

def normalize_text(text):
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def lemmatize_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma:
            lemmas.append(token.lemma)

    return " ".join(lemmas)

def preprocess(text):
    if not isinstance(text, str):
        return ""

    text = remove_after_solution(text)
    text = remove_numbering(text)
    text = normalize_text(text)
    text = lemmatize_text(text)
    return text

# ---------- LOAD DATA ----------

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)
phrases = [item["phrase"] for item in data]
normalized_phrases = [item["normalized"] for item in data]

# ---------- SEARCH ----------

def semantic_search(query, top_k=5):

    normalized_query = preprocess(query)

    print("\nНормализованный запрос:", normalized_query)

    if not normalized_query:
        print("Запрос пуст после обработки")
        return

    query_embedding = model.encode([normalized_query])

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    print("Количество similarity:", len(similarities))

    if len(similarities) == 0:
        print("В базе нет embeddings")
        return

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print("\nTop совпадения:")

    for idx in top_indices:
        print(f"{similarities[idx]:.4f} | {phrases[idx]}")

    max_score = similarities[top_indices[0]]

    print("\nМаксимальный similarity:", round(float(max_score), 4))
# ---------- MAIN ----------


def hybrid_search(query, top_k=5):

    normalized_query = preprocess(query)

    print("\nНормализованный запрос:", normalized_query)

    if not normalized_query:
        print("Запрос пуст после обработки")
        return

    # ---------- EXACT SEARCH ----------
    for item in data:
        if normalized_query in item["normalized"]:
            print("\n EXACT MATCH — найдено совпадение")
            print(item["phrase"])
            return

    # ---------- SEMANTIC SEARCH ----------
    query_embedding = model.encode([normalized_query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    print("\nTop совпадения:")

    for idx in top_indices:
        print(f"{similarities[idx]:.4f} | {phrases[idx]}")

    max_score = similarities[top_indices[0]]

    print("\nМаксимальный similarity:", round(float(max_score), 4))

    # ---------- THRESHOLD ----------
    if max_score > 0.85:
        print(" BLOCK (очень высокая похожесть)")
    elif max_score > 0.75:
        print(" WARNING (подозрительный запрос)")
    else:
        print(" ALLOW (запрос безопасный)")
if __name__ == "__main__":
    while True:
        query = input("\nВведите запрос: ")

        if query.lower() in {"exit", "quit","n"}:
            break

        hybrid_search(query)