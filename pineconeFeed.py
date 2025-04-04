import os
import uuid
import re
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel

# -------------------- Load Environment -------------------- #
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# -------------------- Initialize Tokenizer & Model -------------------- #
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# -------------------- Clean Text -------------------- #
def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- Embedding Function -------------------- #
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled.squeeze().tolist()

# -------------------- Text Splitter -------------------- #
def split_documents(documents, chunk_size=512, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo-0125",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = []
    for doc in documents:
        text = clean_text(doc.page_content)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            new_doc = Document(page_content=chunk, metadata=doc.metadata)
            split_docs.append(new_doc)
    return split_docs

# -------------------- Prepare Vectors -------------------- #
def prepare_pinecone_vectors(docs):
    pinecone_vectors = []
    for doc in docs:
        embedding = get_embedding(doc.page_content)
        
        # Ensure 'text' is included in metadata
        metadata_with_text = dict(doc.metadata)
        metadata_with_text["text"] = doc.page_content

        pinecone_vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata_with_text
        })
    return pinecone_vectors


# -------------------- Push to Pinecone -------------------- #
def push_to_pinecone(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(batch)

# -------------------- Main Pipeline -------------------- #
def main():
    # Step 1: Load CSV
    loader = CSVLoader(
        file_path=r"D:\RAG\rag_file_csv.csv",
        encoding='utf-8',
        source_column="data",  # This should match a column name in your CSV or be omitted
        metadata_columns=["healthcare_type", "name", "description", "remedies"]
    )
    raw_docs = loader.load()
    print(f"[+] Loaded {len(raw_docs)} documents")

    # Step 1.5: Format the page_content manually
    formatted_docs = []
    for doc in raw_docs:
        meta = doc.metadata
        try:
            content = (
                f"{meta['name']} is a {meta['healthcare_type']} condition. "
                f"{meta['description']} Remedies include: {meta['remedies']}."
            )
            formatted_doc = Document(page_content=clean_text(content), metadata=meta)
            formatted_docs.append(formatted_doc)
        except KeyError as e:
            print(f"[!] Missing key: {e}")

    # Step 2: Split documents into chunks
    split_docs = split_documents(formatted_docs)
    print(f"[+] Split into {len(split_docs)} chunks")

    # Step 3: Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "medigraphai"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"[+] Created Pinecone index: {index_name}")
    else:
        print(f"[+] Using existing Pinecone index: {index_name}")

    index = pc.Index(index_name)

    # Step 4: Create and upload vectors
    vectors = prepare_pinecone_vectors(split_docs)
    push_to_pinecone(index, vectors)
    print(f"[✅] Uploaded {len(vectors)} vectors to Pinecone")


if __name__ == "__main__":
    main()
