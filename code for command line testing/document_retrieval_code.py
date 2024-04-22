# import the necessary packages
import torch
import re
import nltk
from sentence_transformers import SentenceTransformer, util


def load_model(model_name):
    """Loads a pre-trained sentence transformer model.

    Args:
        model_name (str): Name of the pre-trained model.

    Returns:
        SentenceTransformer: Loaded model.
    """

    model = SentenceTransformer(model_name)
    return model


def initialize_stopwords(language):
    """Initializes NLTK stopwords for a given language.

    Args:
        language (str): Language for stopwords.

    Returns:
        set: Set of stopwords.
    """

    nltk.download("stopwords")
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words(language))
    return stop_words


def read_documents(directory_path):
    """Reads documents from a directory.

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

    embeddings = load_model.encode(documents, convert_to_tensor=True)
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
        query (str): Query string.
        embedding_dict (dict): Dictionary mapping documents to their embeddings.
        top_n (int, optional): Number of top similar documents to retrieve. Defaults to 5.

    Returns:
        list: List of top N most similar documents.
    """

    query_embedding = load_model.encode(preprocess_text(query), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, torch.stack(list(embedding_dict.values())))[0]
    sorted_indices = torch.argsort(similarities, descending=True)
    top_indices = sorted_indices[:top_n]
    return [list(embedding_dict.keys())[i] for i in top_indices]


if __name__ == "__main__":

    # Initialize stopwords and load model
    stop_words = initialize_stopwords("english")
    load_model = load_model("paraphrase-MiniLM-L6-v2")

    # Read documents and preprocess
    documents = read_documents("data")
    cleaned_documents = []
    for doc in documents:
        cleaned_documents.append(preprocess_text(doc))

    # Generate and store embeddings
    embeddings = generate_embeddings(cleaned_documents)
    embedding_dict = store_embeddings(cleaned_documents, embeddings)

    # Command-Line Interface
    while True:
        query = input("Enter a query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = retrieve_documents(query, embedding_dict)
        print("\nTop 5 most relevant documents to your query:")
        
        # print first 100 characters of each document
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc[:100]}")

