import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSChatInterface:
    def __init__(self):
        self.embeddings_dir = "../embeddings"
        self.index_file = "faiss_index"
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.metadata = []

    def create_or_load_index(self):
        """
        Create a new FAISS index or load an existing one from file.
        """
        if os.path.exists(self.index_file):
            print("Loading FAISS index from file...")
            self.index = faiss.read_index(self.index_file)
            with open(f"{self.index_file}_metadata.json", "r") as f:
                self.metadata = json.load(f)
            print("FAISS index loaded successfully.")
        else:
            print("Creating a new FAISS index...")
            self.index = faiss.IndexFlatL2(384)  
            self.add_embeddings_to_index()
            faiss.write_index(self.index, self.index_file)
            with open(f"{self.index_file}_metadata.json", "w") as f:
                json.dump(self.metadata, f)
            print("FAISS index created and saved.")

    def add_embeddings_to_index(self):
        for file in os.listdir(self.embeddings_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.embeddings_dir, file), "r") as f:
                    data = json.load(f)
                for entry in data:
                    embedding = np.array(entry["embedding"]).astype('float32')
                    self.index.add(np.array([embedding]))  
                    self.metadata.append({
                        "title": entry["title"],
                        "heading": entry["heading"],
                        "chunk": entry["chunk"]
                    })
        print(f"Added {len(self.metadata)} embeddings to the FAISS index.")

    def search(self, query, top_k=2):
        query_embedding = self.model.encode(query).astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1: 
                continue
            results.append({
                "title": self.metadata[idx]["title"],
                "heading": self.metadata[idx]["heading"],
                "chunk": self.metadata[idx]["chunk"],
                "distance": dist
            })
        return results

    def chat(self):
        """
        Chat interface to ask questions and retrieve relevant chunks.
        """
        print("FAISS Chat Interface is ready. Type 'exit' to quit.")
        while True:
            query = input("\nAsk your question: ")
            if query.lower() == 'exit':
                print("Exiting chat interface. Goodbye!")
                break
            results = self.search(query)
            if results:
                print("\nTop Results:")
                for i, result in enumerate(results):
                    print(f"\nResult {i + 1}:")
                    print(f"Title: {result['title']}")
                    print(f"Heading: {result['heading']}")
                    print(f"Chunk: {result['chunk']}")
                    print(f"Distance: {result['distance']:.4f}")
            else:
                print("No relevant chunks found.")

if __name__ == "__main__":
    chat_interface = FAISSChatInterface()
    chat_interface.create_or_load_index()
    chat_interface.chat()
