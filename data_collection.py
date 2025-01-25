import time
import requests
import os
import pdb
import concurrent.futures
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from transformers import GPT2TokenizerFast

from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class DrugDetails:
    record_title: str
    details: Dict[str, str]

class DataCollection:
    def __init__(self):
        self.session = requests.Session()
        self.toc_heading = ["Names and Identifiers", "Drug and Medication Information"]
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


    def token_length(self, text):
        return len(self.tokenizer.encode(text))


    def drug_download(self, drug_id: int) -> Optional[Dict[str, Any]]:
        """Fetch drug information from the PubChem API with retry logic."""
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # New code to read data from local file instead of downloading from API
                url = f"pubchem_data/compound_{drug_id}.json"
                if os.path.exists(url):
                    with open(url) as file:
                        data = json.load(file)
                        return data

                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{drug_id}/JSON/"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                delay = base_delay * (attempt + 1)
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for drug ID {drug_id}: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to fetch drug ID {drug_id} after {max_retries} attempts: {e}")
                    return None
            except ValueError as e:
                print(f"Invalid JSON response for drug ID {drug_id}: {e}")
                return None

    def data_preprocessing(self, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Process raw API data into structured format and apply chunking."""
        if 'Record' not in data:
            return None

        record = data['Record']
        record_title = record.get('RecordTitle', "Unknown Title")

        if record_title == "Unknown Title":
            return None
        
        sections = record.get('Section', [])
        details = {}

        for section in sections:
            heading = section.get("TOCHeading")
            if heading in self.toc_heading:
                extracted_info = self._extract_information(section)
                if extracted_info:
                    details[heading] = [record_title, extracted_info]
        
        if "Drug and Medication Information" in details:
            # Apply chunking to the details text here
            #pdb.set_trace()
            chunked_details = self.apply_chunking(details)
            return chunked_details
        return None
    
    def apply_chunking(self, details: Dict[str, str]) -> List[Dict[str, Any]]:
        """Apply chunking to the extracted details."""
        chunked_data = []
        for heading, text in details.items():
            # Apply the text_chunker (chunking function)
            chunks = self.split_text(text[1])  

            for chunk in chunks:
                chunked_data.append({
                    "title": text[0],
                    "heading": heading,
                    "chunk": chunk,
                })

        return chunked_data
    
    def split_text(self, text: str) -> List[str]:
        """Split text using the Langchain chunking logic."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            length_function=self.token_length,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def _extract_information(self, section: Dict[str, Any]) -> str:
        """Extract and clean information from section."""
        details_list = []
        for sub_section in section.get('Section', []):
            for info in sub_section.get('Information', []):
                if "Value" in info and "StringWithMarkup" in info["Value"]:
                    for detail in info["Value"]["StringWithMarkup"]:
                        text = detail.get('String', '').strip()
                        if text:
                            details_list.append(text)
        return " ".join(details_list)

    def start_process(self, drug_id_start: int, drug_id_limit: int) -> None:
        """Process drug IDs in batches with progress tracking."""
        start_time = time.perf_counter()
        drug_ids = range(drug_id_start, drug_id_limit + 1)
        
        batch_size = 100
        batches = [drug_ids[i:i + batch_size] for i in range(0, len(drug_ids), batch_size)]
        total_batches = len(batches)

        try:
            for batch_num, batch in enumerate(batches, start=1):
                print(f"\nProcessing batch {batch_num}/{total_batches}...")
                processed_drugs = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_id = {executor.submit(self.drug_download, drug_id): drug_id for drug_id in batch}
                    
                    for future in concurrent.futures.as_completed(future_to_id):
                        drug_id = future_to_id[future]
                        try:
                            drug_data = future.result()
                            if drug_data:
                                drug_details = self.data_preprocessing(drug_data)
                                if drug_details:
                                    processed_drugs.append(drug_details)
                        except Exception as e:
                            print(f"Error processing drug ID {drug_id}: {e}")

                # pdb.set_trace()
                """
                Here recieve the chunk of data to push for the next step.
                """
                for drug in processed_drugs:
                    # 2. Calculate embeddings by calling model.encode()
                    embeddings = model.encode(drug)
                    print(embeddings.shape)


        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Cleaning up...")
        finally:
            self.session.close()
            end_time = time.perf_counter()
            print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    obj = DataCollection()
    obj.start_process(drug_id_start=1, drug_id_limit=100)