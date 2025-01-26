import os
import json
import faiss
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


class FAISSChatInterface:
    def __init__(self):
        self.embeddings_dir = "../embeddings"
        self.index_file = "faiss_index"
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.metadata = []
        self.api_token = st.secrets["API_KEY"]
        self.llm_model = 'sophosympatheia/rogue-rose-103b-v0.2:free'
        self.llm_api_url = "https://openrouter.ai/api/v1/chat/completions"

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

    def query_llm(self, query, context=None, chat_history=None):
        """
        Send a query to the LLM with the provided context and chat history.
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": """
                        You are a knowledgeable and conversational medical assistant. Your role is to extract accurate and relevant information about drugs or compounds only from the text provided between <context> and </context> tags. Consider the conversation history provided between <chat_history> and </chat_history> tags to ensure continuity and relevance.

                        When answering the question between <question> and </question> tags:

                             - Respond only if the query is directly related to drugs, compounds, or their medical context, and if the relevant information is present in the context.
                             - Be concise, clear, and conversational.
                             - Avoid making assumptions or fabricating information. Only answer based on the the context, do not add any other information to it.
                             - If the context does not contain relevant information about drugs or compounds, politely respond: "This information is unavailable" ONLY.

                        Respond as if you're having a natural conversation with a person, keeping your tone friendly and professional.
                    """
                },
                {
                    "role": "user",
                    "content": f"<chat_history>{chat_history or ''}</chat_history><context>{context or ''}</context><question>{query}</question>"
                }
            ]
        }
        response = requests.post(self.llm_api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}, {response.text}"

    def chat(self):
        print("FAISS Chat Interface is ready. Type 'exit' to quit.")
        chat_history = []
        while True:
            query = input("\nAsk your question: ")
            if query.lower() == 'exit':
                print("Exiting chat interface. Goodbye!")
                break

            # Check relevance with LLM
            llm_check_response = self.query_llm(query, context="Relevance Check", chat_history=chat_history)
            if llm_check_response.strip().lower() == "this information is unavailable":
                print("\nAgent: This information is unavailable.")
                continue

            # If relevant, search FAISS for context
            results = self.search(query)
            if results:
                context = "\n".join([r["chunk"] for r in results])
                chat_history.append({"role": "assistant", "content": context})
                llm_response = self.query_llm(query, context, chat_history)
                print("\nAgent:")
                print(llm_response)
            else:
                print("\nAgent: No relevant data found.")
                llm_response = self.query_llm(query, context="No context available.", chat_history='')

            # Add to chat history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": llm_response})

def main():
    chat_interface = FAISSChatInterface()
    chat_interface.create_or_load_index()

    st.title('Pharmago')
    st.write('Ask medical-related questions, and I will provide relevant responses.')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm_check_response = chat_interface.query_llm(prompt, context="Relevance Check", chat_history=st.session_state.messages[:5])
                if llm_check_response.strip().lower() == "this information is unavailable":
                    st.markdown("This information is unavailable.")
                else:
                    results = chat_interface.search(prompt)
                    if results:
                        context = "\n".join([f"Title: {r['title']}\nHeading: {r['heading']}\nChunk: {r['chunk']}" for r in results])
                        llm_response = chat_interface.query_llm(prompt, context, st.session_state.messages)
                    else:
                        llm_response = chat_interface.query_llm(prompt, context="No context available.", chat_history=st.session_state.messages)

                    st.markdown(llm_response)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response})

if __name__ == "__main__":
    main()
