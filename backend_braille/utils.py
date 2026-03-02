import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util

def semantic_similarity(correct, student, sbert_model):
    emb1 = sbert_model.encode(correct, convert_to_tensor=True)
    emb2 = sbert_model.encode(student, convert_to_tensor=True)
    return round(float(util.cos_sim(emb1, emb2)) * 100, 2)

def keyword_overlap_score(correct, student):
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
        tfidf = vectorizer.fit_transform([correct, student])
        features = vectorizer.get_feature_names_out()

        correct_words = set(features[i] for i, v in enumerate(tfidf[0].toarray()[0]) if v > 0)
        student_words = set(features[i] for i, v in enumerate(tfidf[1].toarray()[0]) if v > 0)

        if not correct_words:
            return 0.0
        return round(len(correct_words & student_words) / len(correct_words) * 100, 2)
    except:
        return 0.0

def jaccard_similarity(correct, student):
    def tokenize(text):
        text = re.sub(r'[^a-z\s]', '', text.lower())
        return set(text.split())

    a, b = tokenize(correct), tokenize(student)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b) * 100, 2)

def length_penalty(correct, student):
    r = len(student.split()) / max(len(correct.split()), 1)
    if 0.5 <= r <= 1.5: return 1.0
    elif r < 0.8: return 0.6
    return 0.95