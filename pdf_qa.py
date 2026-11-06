
import os
import pandas as pd
import chromadb
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(BASE_DIR, "dataset", "sample.pdf")
TEXT_PATH = os.path.join(BASE_DIR, "dataset", "extracted_text.txt")
CHUNKS_PATH = os.path.join(BASE_DIR, "dataset", "chunks.csv")
QUERIES_PATH = os.path.join(BASE_DIR, "dataset", "queries.csv")
ANSWERS_PATH = os.path.join(BASE_DIR, "dataset", "answers.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"


# ==================================================
# 2. EXTRACT TEXT FROM PDF
# ==================================================

def extract_pdf_text():

    reader = PdfReader(PDF_PATH)
    extracted_text = ""

    print("\nTotal PDF Pages:", len(reader.pages))

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            extracted_text += text + "\n"

        print("Extracted Page:", page_number + 1)

    return extracted_text


# ==================================================
# 3. SAVE EXTRACTED TEXT
# ==================================================

def save_extracted_text(text):

    with open(TEXT_PATH, "w", encoding="utf-8") as file:
        file.write(text)

    print("\nExtracted text saved successfully!")


# ==================================================
# 4. SPLIT TEXT INTO CHUNKS
# ==================================================

def split_text_into_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    print("\nTotal Chunks:", len(chunks))

    return chunks


# ==================================================
# 5. SAVE CHUNKS INTO CSV
# ==================================================

def save_chunks_to_csv(chunks):

    df = pd.DataFrame({
        "chunk_id": range(1, len(chunks) + 1),
        "text": chunks
    })

    df.to_csv(CHUNKS_PATH, index=False, encoding="utf-8")

    print("Chunks saved successfully!")


# ==================================================
# 6. GENERATE VECTOR EMBEDDINGS
# ==================================================

def create_embeddings(chunks):

    print("\nLoading Sentence Transformer...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("\nEmbeddings generated successfully!")
    print("Embedding Shape:", embeddings.shape)

    return model, embeddings


# ==================================================
# 7. STORE EMBEDDINGS IN CHROMADB
# ==================================================

def store_embeddings(chunks, embeddings):

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection_name = "pdf_documents"

    existing = [
        collection.name
        for collection in client.list_collections()
    ]

    if collection_name in existing:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(
        name=collection_name
    )

    ids = [f"chunk_{i+1}" for i in range(len(chunks))]

    metadatas = [
        {"chunk_id": i+1}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\nChromaDB storage successful!")
    print("Stored Documents:", collection.count())

    return collection


# ==================================================
# 8. RETRIEVE TOP-5 RELEVANT CHUNKS
# ==================================================

def retrieve_relevant_chunks(query, model, collection):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, collection.count()),
        include=["documents", "distances"]
    )

    retrieved_chunks = results["documents"][0]

    retrieved_ids = results["ids"][0]

    print("\nTop-5 Retrieved Chunks:")

    for i, chunk in enumerate(retrieved_chunks):

        print(f"\nChunk {i+1}: {retrieved_ids[i]}")
        print(chunk[:250])

    return retrieved_chunks


# ==================================================
# 9. GENERATE ANSWER WITH RAG
# ==================================================

def generate_rag_answer(query, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a PDF Question Answering Assistant.

Answer the question using ONLY the PDF context.

Do not use outside knowledge.
Do not invent information.

If the answer is unavailable in the context,
say:
"The answer is not available in the PDF."

PDF CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"]


# ==================================================
# 10. GENERATE ANSWER WITHOUT RAG
# ==================================================

def generate_non_rag_answer(query):

    prompt = f"""
Answer the following question using your general
knowledge. No PDF context is provided.

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"]


# ==================================================
# 11. PROCESS ALL 15 QUESTIONS
# ==================================================

def process_all_questions(model, collection):

    if not os.path.exists(QUERIES_PATH):
        print("\nERROR: queries.csv not found!")
        return

    queries_df = pd.read_csv(QUERIES_PATH)

    if not {"query_id", "question"}.issubset(queries_df.columns):
        print("\nERROR: queries.csv must contain query_id and question columns.")
        return

    if queries_df.empty:
        print("\nERROR: No questions found!")
        return

    answers_data = []

    total_questions = len(queries_df)

    print("\n======================================")
    print(" AUTOMATED PDF QUESTION ANSWERING")
    print("======================================")

    for index, row in queries_df.iterrows():

        query_id = row["query_id"]
        question = str(row["question"]).strip()

        if not question:
            print(f"\nSkipping empty question: {query_id}")
            continue

        print("\n--------------------------------------")
        print(f"Question {index + 1}/{total_questions}")
        print("ID:", query_id)
        print("Question:", question)

        try:

            # Retrieve relevant PDF chunks
            retrieved_chunks = retrieve_relevant_chunks(
                question,
                model,
                collection
            )

            # Generate answer with RAG
            print("\nGenerating RAG answer...")

            rag_answer = generate_rag_answer(
                question,
                retrieved_chunks
            )

            # Generate answer without RAG
            print("\nGenerating answer WITHOUT RAG...")

            non_rag_answer = generate_non_rag_answer(
                question
            )

            # Save results
            answers_data.append({

                "query_id": query_id,

                "question": question,

                "retrieved_chunks": "\n\n".join(
                    retrieved_chunks
                ),

                "rag_answer": rag_answer,

                "non_rag_answer": non_rag_answer

            })

            print("\nRAG ANSWER:")
            print(rag_answer)

            print("\nNON-RAG ANSWER:")
            print(non_rag_answer)

            print("\nQuestion completed successfully!")

        except Exception as e:

            print("\nError processing", query_id, ":", e)

            answers_data.append({

                "query_id": query_id,
                "question": question,
                "retrieved_chunks": "",
                "rag_answer": "Error: " + str(e),
                "non_rag_answer": ""

            })

    # Save all answers into CSV
    answers_df = pd.DataFrame(answers_data)

    answers_df.to_csv(
        ANSWERS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print(" WEEK 13 - AUTOMATED EVALUATION DONE!")
    print("======================================")

    print("\nTotal Questions:", total_questions)
    print("Successfully Processed:", len(answers_data))
    print("Answers saved to:", ANSWERS_PATH)


# ==================================================
# 12. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print("\n======================================")
    print("       PDF QUESTION ANSWERING SYSTEM")
    print("======================================")

    try:

        if not os.path.exists(PDF_PATH):
            print("\nERROR: sample.pdf not found!")

        elif not os.path.exists(QUERIES_PATH):
            print("\nERROR: queries.csv not found!")

        else:

            # Extract PDF text
            text = extract_pdf_text()

            if not text.strip():
                print("\nERROR: No readable text found in PDF!")

            else:

                # Save extracted text
                save_extracted_text(text)

                # Split text into chunks
                chunks = split_text_into_chunks(text)

                if not chunks:
                    print("\nERROR: No chunks created!")

                else:

                    # Save chunks
                    save_chunks_to_csv(chunks)

                    # Generate embeddings
                    model, embeddings = create_embeddings(chunks)

                    # Store embeddings
                    collection = store_embeddings(
                        chunks,
                        embeddings
                    )

                    # Process all questions
                    process_all_questions(
                        model,
                        collection
                    )

    except Exception as e:

        print("\nPROGRAM ERROR:", e)
