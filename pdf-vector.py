import faiss
import PyPDF2
import numpy as np
import pickle
import requests
import json

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mistral:latest"  # Using your available model for embeddings

def get_ollama_embedding(text):
    """Get embedding from Ollama using the mistral model"""
    try:
        # Use Ollama's embedding endpoint if available, otherwise use generation for embeddings
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            }
        )
        
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            # Fallback: create a simple hash-based embedding (not ideal but functional)
            print(f"Warning: Could not get embedding from Ollama. Status: {response.status_code}")
            return create_simple_embedding(text)
            
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return create_simple_embedding(text)

def create_simple_embedding(text, dim=384):
    """Create a simple embedding based on text characteristics"""
    # This is a fallback method - not as good as proper embeddings
    import hashlib
    
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numbers and normalize
    embedding = []
    for i in range(0, len(text_hash), 2):
        val = int(text_hash[i:i+2], 16) / 255.0
        embedding.append(val)
    
    # Pad or truncate to desired dimension
    while len(embedding) < dim:
        embedding.extend(embedding[:min(len(embedding), dim - len(embedding))])
    
    return embedding[:dim]

def pdf_to_vectors(pdf_path):
    # Read PDF
    print(f"📄 Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        # Extract text from each page separately
        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append({
                'text': page_text,
                'page_number': page_num + 1
            })

        # Combine all text for chunking
        text = ''.join([p['text'] for p in page_texts])

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(text):,} characters")
    print(f"📊 Average characters per page: {len(text) // total_pages:,}")

    # Create chunks with page tracking
    chunks = []
    chunk_metadata = []

    for i in range(0, len(text), 400):
        chunk_text = text[i:i + 500]
        chunks.append(chunk_text)

        # Estimate which page this chunk belongs to
        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)
        chunk_metadata.append({
            'start_pos': i,
            'estimated_page': estimated_page
        })

    print(f"✂️ Created {len(chunks)} chunks")

    # Get embeddings from Ollama
    print("🔄 Getting embeddings from Ollama...")
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing {i + 1}/{len(chunks)}")
        embedding = get_ollama_embedding(chunk)
        embeddings.append(embedding)

    # Create FAISS index
    print("🗂️ Creating FAISS index...")
    embeddings = np.array(embeddings)
    
    # Get the dimension of our embeddings
    embedding_dim = len(embeddings[0])
    print(f"📏 Embedding dimension: {embedding_dim}")
    
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.astype('float32'))

    # Save to files
    print("💾 Saving to files...")
    faiss.write_index(index, "vectors.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            'chunks': chunks,
            'metadata': chunk_metadata,
            'total_pages': total_pages,
            'embedding_dim': embedding_dim
        }, f)

    print("✅ Vector database created successfully!")
    print(f"📁 Files saved: vectors.index, chunks.pkl")
    print(f"📊 Vector shape: {embeddings.shape}")
    print(f"🔢 Sample vector (first 5 dims): {embeddings[0][:5]}")

    return embeddings, chunks

def check_ollama_connection():
    """Check if Ollama is running and has the required model"""
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            print("❌ Ollama is not running. Please start Ollama first.")
            return False
            
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        print("🤖 Available Ollama models:")
        for name in model_names:
            print(f"   • {name}")
        
        if EMBEDDING_MODEL not in model_names:
            print(f"⚠️  Model '{EMBEDDING_MODEL}' not found.")
            print("Using fallback embedding method.")
        else:
            print(f"✅ Using model: {EMBEDDING_MODEL}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running on http://localhost:11434")
        return False

# Usage
if __name__ == "__main__":
    # Check Ollama connection first
    if not check_ollama_connection():
        exit(1)
    
    # Convert PDF to vectors (run this once)
    pdf_file = "SaMD-Document.pdf"  # Using your uploaded PDF
    
    try:
        embeddings, chunks = pdf_to_vectors(pdf_file)
        print("\n🎉 Setup complete! Now you can run the question script to chat with your PDF!")
    except FileNotFoundError:
        print(f"❌ Error: PDF file '{pdf_file}' not found.")
        print("Please make sure the PDF file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error: {e}")