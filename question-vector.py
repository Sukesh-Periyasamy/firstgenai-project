import faiss
import numpy as np
import pickle
import os
import requests
import json
import hashlib

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mistral:latest"  # For embeddings
CHAT_MODEL = "llama3:8b"  # For generating answers - using your best model

def get_ollama_embedding(text):
    """Get embedding from Ollama using the mistral model"""
    try:
        # Use Ollama's embedding endpoint if available
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
            # Fallback: create a simple hash-based embedding
            return create_simple_embedding(text)
            
    except Exception as e:
        print(f"Warning: Could not get embedding from Ollama: {e}")
        return create_simple_embedding(text)

def create_simple_embedding(text, dim=384):
    """Create a simple embedding based on text characteristics"""
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

def get_ollama_response(prompt):
    """Get response from Ollama chat model"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: Could not get response from Ollama (Status: {response.status_code})"
            
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def ask_question(question):
    # Check if vector files exist
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Error: Vector database not found!")
        print("🔧 Please run the PDF processing script first to create the database.")
        return None

    try:
        # Load saved data
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']
        embedding_dim = data.get('embedding_dim', 384)

        # Get question embedding
        query_vector = get_ollama_embedding(question)
        
        # Ensure the embedding has the right dimension
        if len(query_vector) != embedding_dim:
            query_vector = query_vector[:embedding_dim] if len(query_vector) > embedding_dim else query_vector + [0] * (embedding_dim - len(query_vector))
        
        query_vector = np.array(query_vector).reshape(1, -1)

        # Search similar chunks
        scores, indices = index.search(query_vector.astype('float32'), 3)

        # Show similarity scores and page info for debugging
        print(f"🔍 Found {len(indices[0])} relevant chunks:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            page_num = metadata[idx]['estimated_page']
            print(f"   Chunk {i + 1}: Score {score:.3f} (≈Page {page_num})")

        # Build context with page information
        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page_num = metadata[idx]['estimated_page']
            context_parts.append(f"[Page {page_num}]: {chunk_text}")

        context = '\n\n'.join(context_parts)

        # Create a comprehensive prompt for the local model
        prompt = f"""You are an AI assistant helping to answer questions about a {total_pages}-page document about SaMD (Software as Medical Device) regulations. 

Context from the document:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. When relevant, mention specific page numbers from the document. If the context doesn't contain enough information to answer the question, say so clearly.

Answer:"""

        # Get answer from local Ollama model
        print("🤖 Generating response using local model...")
        response = get_ollama_response(prompt)
        return response

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        return None

def check_ollama_connection():
    global CHAT_MODEL  # Move this to the beginning of the function
    """Check if Ollama is running and has the required models"""
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
        
        # Check for required models
        missing_models = []
        if EMBEDDING_MODEL not in model_names:
            missing_models.append(EMBEDDING_MODEL)
        if CHAT_MODEL not in model_names:
            missing_models.append(CHAT_MODEL)
            
        if missing_models:
            print(f"⚠️  Missing models: {', '.join(missing_models)}")
            if CHAT_MODEL not in model_names:
                # Suggest alternative chat models
                available_chat_models = [name for name in model_names if any(x in name for x in ['llama', 'mistral', 'gemma'])]
                if available_chat_models:
                    print(f"💡 Consider using one of these available models: {', '.join(available_chat_models)}")
                    CHAT_MODEL = available_chat_models[0]  # Use the first available model
                    print(f"🔄 Switched to using: {CHAT_MODEL}")
        else:
            print(f"✅ Using embedding model: {EMBEDDING_MODEL}")
            print(f"✅ Using chat model: {CHAT_MODEL}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running on http://localhost:11434")
        return False

def main():
    # Check Ollama connection first
    if not check_ollama_connection():
        return

    # Check if database exists
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector database not found!")
        print("🔧 Please run the PDF processing script first to create the database.")
        print("📋 Steps:")
        print("   1. Run: python pdf_to_vectors_local.py")
        print("   2. Then run: python question_answering_local.py")
        return

    # Load database info
    try:
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        total_pages = data['total_pages']

        print(f"✅ Database loaded: {len(chunks)} chunks from {total_pages} pages")
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")
        return

    # Interactive question loop
    print("\n" + "=" * 60)
    print("🤖 Local RAG System Ready! Ask me questions about your SaMD PDF")
    print("💡 Type 'bye', 'quit', 'exit', or 'q' to exit")
    print("📊 Type 'info' to see database statistics")
    print("🔧 Type 'models' to see available Ollama models")
    print("=" * 60)

    while True:
        question = input("\n❓ Your question: ").strip()

        # Check for exit commands
        if question.lower() in ['bye', 'quit', 'exit', 'q']:
            print("👋 Goodbye! Thanks for using the local RAG system!")
            break

        # Show database info
        if question.lower() == 'info':
            print(f"📊 Database Info:")
            print(f"   • Total pages: {total_pages}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Vector dimensions: {data.get('embedding_dim', 'Unknown')}")
            print(f"   • Average chunks per page: {len(chunks) / total_pages:.1f}")
            print(f"   • Sample chunk: {chunks[0][:100]}...")
            continue

        # Show model info
        if question.lower() == 'models':
            check_ollama_connection()
            continue

        # Skip empty questions
        if not question:  
            print("⚠️ Please enter a question!")
            continue

        print("🔍 Searching and generating answer...")
        answer = ask_question(question)

        if answer:
            print(f"🤖 Answer: {answer}")
        else:
            print("❌ Sorry, I couldn't generate an answer. Please try a different question.")

if __name__ == "__main__":
    main()