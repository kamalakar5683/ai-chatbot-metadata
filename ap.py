import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.vectorstores import FAISS as LangChainFAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# Function to extract text from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text from paragraphs or any other tags you deem necessary
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return text

# Load the Llama 3.2 model via Ollama (for chat model and embeddings)
ollama_model = Ollama(model="llama3.2", temperature=0.3)  # Set lower temperature for direct responses
embeddings_model = OllamaEmbeddings(model="llama3.2")

# FAISS Index for Memory Management
embedding_example = embeddings_model.embed_query("Example query to check embedding size")
embedding_dimension = len(embedding_example)

index = faiss.IndexFlatL2(embedding_dimension)  # Use the correct embedding dimension
docstore = InMemoryDocstore()  # Use InMemoryDocstore for storing documents
index_to_docstore_id = {}

vectorstore = LangChainFAISS(embedding_function=embeddings_model.embed_query, 
                             index=index, 
                             docstore=docstore, 
                             index_to_docstore_id=index_to_docstore_id)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 'k' is the number of previous context messages
memory = VectorStoreRetrieverMemory(retriever=retriever)

conversation_chain = ConversationChain(
    llm=ollama_model, 
    memory=memory
)

# Streamlit UI
st.title("AI-Powered Chat Application with Web Data")

# Input Field for URL and User Query
url_input = st.text_input("Enter URL to fetch data from:")
user_input = st.text_input("Enter your query:")

if url_input:
    # Step 1: Extract data from the URL
    page_text = extract_text_from_url(url_input)
    
    # Step 2: Embed the extracted data and store it in FAISS
    embedded_data = embeddings_model.embed_query(page_text)
    vectorstore.add_texts([page_text])  # Add the embedded data to the vectorstore
    
    # Display confirmation of data extraction
    st.write("Data successfully fetched from the URL and added to the knowledge base.")

if user_input:
    # Step 3: Generate a more contextually aware response
    prompt = f"Answer the following query using the knowledge from the URL data. Query: {user_input}"
    response = conversation_chain.predict(input=prompt)

    # Step 4: Save User Input and AI Response into FAISS Vectorstore
    vectorstore.add_texts([user_input, response])  # Add both user input and response for context

    # Display the AI response
    st.write("AI Response: ", response)

    # Step 5: Display Relevant Past Conversations
    st.write("Relevant Past Conversations:")
    relevant_interactions = vectorstore.similarity_search(user_input, k=5)

    # Filter out irrelevant responses based on similarity threshold
    if relevant_interactions:
        for i, interaction in enumerate(relevant_interactions):
            with st.expander(f"Interaction {i+1}"):
                st.write(f"User Input: {interaction.page_content}")  # Show user input
                ai_response = interaction.metadata.get('response', 'No response saved')
                st.write(f"AI Response: {ai_response}")  # Show AI response
    else:
        st.write("No relevant past conversations found.")

# Optionally, allow users to clear the chat history
if st.button("Clear Conversation History"):
    vectorstore.clear()
    st.write("Conversation history cleared.")