import pandas as pd
import re
import json
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from sentence_transformers import SentenceTransformer

CSV_PATH = "exportfsm.csv"
OUTPUT_PATH = "prepared_blacklist.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------- NATASHA INIT ----------

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

# ---------- EMBEDDING MODEL ----------

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
    text = remove_after_solution(text)
    text = remove_numbering(text)
    text = normalize_text(text)
    text = lemmatize_text(text)
    return text


# ---------- MAIN ----------

def main():

    df = pd.read_csv(
        CSV_PATH,
        encoding="cp1251",
        sep=";"
    )

    phrases = []
    normalized_texts = []

  
    for _, row in df.iterrows():
        phrase = row.get("Материал")

        if pd.isna(phrase):
            continue

        phrase = str(phrase)
        normalized = preprocess(phrase)

        phrases.append(phrase)
        normalized_texts.append(normalized)

    print("Генерация embeddings батчами...")

    
    embeddings = model.encode(
        normalized_texts,
        batch_size=64,           # можно 32–128
        show_progress_bar=True
    )

   
    processed_data = []

    for i in range(len(phrases)):
        processed_data.append({
            "phrase": phrases[i],
            "normalized": normalized_texts[i],
            "category": "illegal",
            "severity": 3,
            "embedding": embeddings[i].tolist()
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print("Готово!")


if __name__ == "__main__":
    main()