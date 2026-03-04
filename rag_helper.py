"""
RAG Helper for ATI.SU Knowledge Base
Usage:
    from rag_helper import RAGSearch
    rag = RAGSearch("knowledge_base_rag.json")
    results = rag.search("Как рассчитывается индекс ATI.SU?", top_k=5)
    for r in results:
        print(r["score"], r["section"], r["text"][:200])
"""
import json, re, math
from collections import Counter

class RAGSearch:
    def __init__(self, json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.df = data["index"]["df"]
        self.N = data["index"]["N"]

    def tokenize(self, text):
        return re.findall(r'[а-яёa-z0-9]+', text.lower())

    def tfidf(self, tokens):
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            if term in self.df:
                idf = math.log((self.N + 1) / (self.df[term] + 1)) + 1
                vec[term] = (count / max(len(tokens), 1)) * idf
        return vec

    def cosine(self, v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        n1 = math.sqrt(sum(x*x for x in v1.values()))
        n2 = math.sqrt(sum(x*x for x in v2.values()))
        return dot / (n1 * n2) if n1 and n2 else 0.0

    def search(self, query, top_k=5):
        q_vec = self.tfidf(self.tokenize(query))
        scored = []
        for chunk in self.chunks:
            score = self.cosine(q_vec, chunk["tfidf"])
            scored.append({"score": round(score, 4), **chunk})
        return sorted(scored, key=lambda x: -x["score"])[:top_k]

    def get_context(self, query, top_k=5):
        """Returns formatted context string for LLM prompt"""
        results = self.search(query, top_k)
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Источник {i} | {r['section']}]\n{r['text']}")
        return "\n\n".join(parts)
