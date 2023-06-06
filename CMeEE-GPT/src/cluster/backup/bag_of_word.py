from src.utils.settings import get_json_data
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer

class Bag_shots:
    def __init__(self):
        self.corpus = []
        self.data = get_json_data("train")
        for i in range(len(self.data)):
            text = self.data[i]["text"]
            self.corpus.append(text)
            # words = list(jieba.cut(text, cut_all=True))
        # self.vectorizer = HashingVectorizer(n_features = 10, norm = "l2")
        tokenizer = AutoTokenizer.from_pretrained('mc-bert-base', use_fast=False)
        self.vectorizer = CountVectorizer(tokenizer = tokenizer)
        self.features = self.vectorizer.fit_transform(self.corpus)

    def get_shots(self, text, nums):
        text_features = self.vectorizer.transform([text])
        similarities = cosine_similarity(text_features, self.features)[0]
        most_similar_indices = np.argsort(similarities)[-nums:]
        
        # most_similar_texts = [self.data[i] for i in most_similar_indices]
        most_similar_texts = [self.corpus[i] for i in most_similar_indices]
        similarity_scores = similarities[most_similar_indices]
        for i in range(nums):
            print(similarity_scores[i], most_similar_texts[i])
        return most_similar_texts
        