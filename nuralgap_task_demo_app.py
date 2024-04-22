import streamlit as st
import torch
import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model
model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Initialize NLTK stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Preprocess text by removing stopwords and converting to lowercase
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Generate embeddings for a set of documents
def generate_embeddings(documents):
    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings

# Store document embeddings in an in-memory data structure (dictionary)
def store_embeddings(documents, embeddings):
    embedding_dict = {}
    for i, doc in enumerate(documents):
        embedding_dict[doc] = embeddings[i]
    return embedding_dict

# Retrieve top N most similar documents based on cosine similarity
def retrieve_documents(query, embedding_dict, top_n=5):
    query_embedding = model.encode(preprocess_text(query), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, torch.stack(list(embedding_dict.values())))[0]
    sorted_indices = torch.argsort(similarities, descending=True)
    top_indices = sorted_indices[:top_n]
    return [list(embedding_dict.keys())[i] for i in top_indices]

# Streamlit app to retrieve relevant documents based on user queries
def main():
    st.title("Document Retrieval App")
    st.write("Enter a query to retrieve relevant documents:")

    documents = [
        "Document 1: This is the first document about sentence embeddings.",
        "Document 2: Sentence embeddings are useful for natural language processing.",
        "Document 3: Embeddings can capture semantic meaning in text.",
        "Document 4: Pre-trained models like BERT and MiniLM are commonly used for embeddings.",
    ]
    st.text_area("Documents", "\n".join(documents))
    # User input
    query = st.text_input("Enter query:")

    if query:
        embeddings = generate_embeddings(documents)
        embedding_dict = store_embeddings(documents, embeddings)
        results = retrieve_documents(query, embedding_dict)

        st.text("\nTop 3 most relevant documents:")
        for i, doc in enumerate(results):
            if i < 3:
                st.text(f"{i+1}. {doc}")

if __name__ == "__main__":
    main()