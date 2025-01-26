import json
import os
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self):
        self.input_file = "flattened_chunks.json"
        self.output_dir = "../embeddings"  
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Load the preprocessed data from a JSON file.
        """
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} records from {self.input_file}.")

    def generate_embeddings(self):
        """
        Generate embeddings for the loaded data and save them in separate files per drug.
        """
        drug_data = {}

        # Group chunks by drug title
        for item in self.data:
            title = item["title"]
            chunk = item["chunk"]
            embedding = self.model.encode(chunk).tolist()

            if title not in drug_data:
                drug_data[title] = []

            drug_data[title].append({
                "title": title,
                "heading": item["heading"],
                "chunk": chunk,
                "embedding": embedding
            })

        for title, chunks in drug_data.items():
            safe_title = title.replace(" ", "_").replace("/", "_") 
            file_path = os.path.join(self.output_dir, f"{safe_title}.json")
            
            with open(file_path, 'w') as f:
                json.dump(chunks, f, indent=4)
            print(f"Saved {len(chunks)} chunks for drug '{title}' to {file_path}.")

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.load_data()
    generator.generate_embeddings()
