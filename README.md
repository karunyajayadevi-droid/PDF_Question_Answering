# Week 13 – PDF Question Answering System

## Project Title
Building a PDF Question Answering System using Retrieval-Augmented Generation (RAG).

## 1. Introduction
This project develops an AI-powered PDF Question Answering System. It allows users to ask questions about PDF documents and receive relevant answers using a locally running Large Language Model.

The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from PDF documents before generating answers.

## 2. Objectives
- Extract text from PDF documents.
- Divide extracted text into smaller chunks.
- Generate vector embeddings.
- Store embeddings in ChromaDB.
- Retrieve the Top-5 relevant chunks.
- Generate answers using Ollama Llama 3.2.
- Compare RAG and Non-RAG answers.
- Automate evaluation using 15 questions.

## 3. Technologies Used
- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2
- Pandas
- PyPDF
- Jupyter Notebook
- Visual Studio Code

## 4. Models Used

**Embedding Model:** all-MiniLM-L6-v2

**Language Model:** Llama 3.2

## 5. Project Folder Structure

```text
PDF_Question_Answering/
│
├── dataset/
│   ├── sample.pdf
│   ├── extracted_text.txt
│   ├── chunks.csv
│   ├── queries.csv
│   └── answers.csv
│
├── chroma_db/
│
├── notebook/
│   └── Week13_PDF_Question_Answering.ipynb
│
├── venv/
│
├── pdf_qa.py
├── requirements.txt
└── README.md
```

## 6. Project Workflow

1. Load the PDF document.
2. Extract text using PyPDF.
3. Split the text into smaller chunks using LangChain.
4. Generate vector embeddings using Sentence Transformers.
5. Store embeddings in ChromaDB.
6. Retrieve the Top-5 relevant chunks for each question.
7. Generate RAG answers using Ollama Llama 3.2.
8. Generate Non-RAG answers without PDF context.
9. Compare both answers.
10. Save the results in answers.csv.

## 7. Installation and Setup

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Install Ollama and download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

## 8. How to Run the Project

Activate the Python virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Run the Python program:

```bash
python pdf_qa.py
```

The program processes the questions and generates answers automatically.

## 9. Output Files

- extracted_text.txt – Contains extracted PDF text.
- chunks.csv – Contains divided text chunks.
- queries.csv – Contains 15 questions.
- answers.csv – Contains retrieved chunks, RAG answers and Non-RAG answers.

## 10. RAG vs Non-RAG Comparison

**RAG:** Generates answers using relevant information retrieved from the PDF document.

**Non-RAG:** Generates answers directly using the language model without PDF context.

This comparison helps understand the usefulness of document retrieval in question answering.

## 11. Conclusion

The PDF Question Answering System demonstrates how Retrieval-Augmented Generation can be used to answer questions from PDF documents.

By combining LangChain, ChromaDB, Sentence Transformers and Ollama Llama 3.2, the project provides a local AI-based document question answering solution.

The project also demonstrates automated evaluation by comparing RAG and Non-RAG answers for 15 questions.