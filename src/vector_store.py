import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleVectorStore:
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        embeddings = self.model.encode(chunks)
        self.embeddings = np.array(embeddings).astype("float32")

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def similarity_search(self, query, k=3):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results


def create_vector_store(chunks):
    return SimpleVectorStore(chunks)