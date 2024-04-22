# streamlit run nuralgap_task_demo_app.py
import streamlit as st
import torch
import re
import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords


model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
stop_words = set(stopwords.words("english"))


def read_documents(directory_path):
    """Loads document content from a directory.

    Args:
        directory_path (str): Path to the directory containing documents.

    Returns:
        list: List of documents as strings.
    """

    documents = []
    for i in range(1, 11):
        with open(f"{directory_path}/doc_{i}.txt", "r", encoding="utf-8") as file:
            content = file.read()
            clean_content = content.replace("\n", "")
            documents.append(clean_content)
    return documents


def preprocess_text(text):
    """Preprocesses text by removing stopwords and converting to lowercase.

    Args:
        text (str): Text to preprocess.

    Returns:
        str: Preprocessed text.
    """

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text


def generate_embeddings(documents):
    """Generates embeddings for a set of documents.

    Args:
        documents (list): List of documents as strings.

    Returns:
        torch.Tensor: Tensor of document embeddings.
    """

    embeddings = model.encode(documents, convert_to_tensor=True)
    return embeddings


def store_embeddings(documents, embeddings):
    """Stores document embeddings in a dictionary.

    Args:
        documents (list): List of documents as strings.
        embeddings (torch.Tensor): Tensor of document embeddings.

    Returns:
        dict: Dictionary mapping documents to their embeddings.
    """

    embedding_dict = {}
    for i, doc in enumerate(documents):
        embedding_dict[doc] = embeddings[i]
    return embedding_dict


def retrieve_documents(query, embedding_dict, top_n=5):
    """Retrieves top N most similar documents based on cosine similarity.

    Args:
        query (str): User query.
        embedding_dict (dict): Dictionary mapping documents to their embeddings.
        top_n (int, optional): Number of top similar documents to retrieve. Defaults to 5.

    Returns:
        list: List of top N most similar documents.
    """

    query_embedding = model.encode(preprocess_text(query), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, torch.stack(list(embedding_dict.values())))[0]
    sorted_indices = torch.argsort(similarities, descending=True)
    top_indices = sorted_indices[:top_n]
    return [list(embedding_dict.keys())[i] for i in top_indices]


def main():
    """Streamlit app to retrieve relevant documents based on user queries."""

    st.title("Document Retrieval App ")
    st.subheader("Neuralgap NLP Task")
    url = "https://github.com/virajpvs/Neuralgap-NLP-Task"
    text_with_link = f"View the code on repo [Neuralgap-NLP-Task]({url})"
    st.write(text_with_link)
    st.divider()

    st.markdown("#### This app retrieves most relevant documents based on user queries.")
    st.divider()
    documents = read_documents("data")
    st.write("Check out the documents available for retrieval:")
    st.text_area("Documents", "\n".join(documents))

    # User input
    query = st.text_input("Enter a query to retrieve relevant documents:")


    cleaned_documents = []
    for doc in documents:
        cleaned_documents.append(preprocess_text(doc))

    embeddings = generate_embeddings(cleaned_documents)
    embedding_dict = store_embeddings(cleaned_documents, embeddings)
    results = retrieve_documents(query, embedding_dict)

    if query:
        st.divider()

        st.markdown("### Top 5 relevant documents to your query:")
        for i, doc in enumerate(results):
            if i < 5:
                st.text(f"{i+1}. {doc[:100]}")


# if __name__ == "__main__": 
main()
