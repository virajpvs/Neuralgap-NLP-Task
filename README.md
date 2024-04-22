# Document Retrieval App with Streamlit and Sentence Transformers (Neuralgap-NLP-Task)

## Test the Document Retrieval App

https://neuralgap-nlp-task-domo.streamlit.app/

This project implements a web application using Streamlit for document retrieval based on semantic similarity. It leverages the pre-trained sentence-transformer model `paraphrase-MiniLM-L6-v2` for effective text embedding and retrieval.

## Requirements

- Python 3.11 (highly recommended)
- Streamlit (`pip install streamlit`)
- Sentence Transformers (`pip install sentence-transformers`)
- Torch (`pip install torch`) (consider using a GPU-compatible version for faster processing)
- NLTK (`pip install nltk`)

## Installation and Setup

**1. Create a virtual environment (recommended for dependency isolation):**  
Creating a virtual environment helps isolate project dependencies and avoid conflicts with other Python installations on your system. Here's how to do it on Windows:

**a. Open Command Prompt:**

Press the Windows key, type "cmd", and press Enter.

**b. Create the virtual environment:**  
 `   python -m venv venv # Replace 'venv' with your desired virtual environment name
  `

This command creates a directory named `venv` in your current working directory. This directory contains the isolated Python environment and its packages.

**c. Activate the virtual environment:**  
 `   venv\Scripts\activate.bat  # Activate the virtual environment
  `

Now, your command prompt will indicate that the virtual environment is active (usually denoted by `(venv)` before the prompt). This means you're working within the isolated environment and any packages installed here won't affect your system-wide Python installation.

**2. Install dependencies:**  
Once your virtual environment is activated, proceed with installing the required packages:

```
pip install -r requirements.txt

```

**3. Running the App:**  
**a. Navigate to the project directory:**

```
cd Neuralgap-NLP-Task
```

**b. Run the application:**

```
streamlit run nuralgap_task_demo_app.py
```
