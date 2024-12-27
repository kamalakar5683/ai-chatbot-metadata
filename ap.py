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
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text if text.strip() else "No content found."
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

# Initialize Llama 3.2 model and embeddings
ollama_model = Ollama(model="llama3.2", temperature=0.3)
embeddings_model = OllamaEmbeddings(model="llama3.2")

# Initialize FAISS VectorStore
embedding_example = embeddings_model.embed_query("Example query to check embedding size")
embedding_dimension = len(embedding_example)
index = faiss.IndexFlatL2(embedding_dimension)
docstore = InMemoryDocstore()
index_to_docstore_id = {}

vectorstore = LangChainFAISS(
    embedding_function=embeddings_model.embed_query,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=retriever)
conversation_chain = ConversationChain(llm=ollama_model, memory=memory)

# Streamlit UI
st.title("AI-Powered Chat Application with Web Data")

# Input fields for URL and query
url_input = st.text_input("Enter a URL to fetch data:")
user_input = st.text_input("Enter your query:")

if url_input:
    st.info("Fetching and embedding content from the URL...")
    page_text = extract_text_from_url(url_input)
    if page_text:
        vectorstore.add_texts([page_text], metadata=[{"source": url_input}])
        st.success("Data successfully added to the knowledge base.")

if user_input:
    st.info("Generating AI response...")
    prompt = f"Answer the following query using the knowledge from the URL data. Query: {user_input}"
    response = conversation_chain.predict(input=prompt)
    vectorstore.add_texts([user_input, response], metadata=[{"type": "user_query"}, {"type": "ai_response"}])
    st.write("AI Response:", response)

    # Display relevant past conversations
    st.write("Relevant Past Conversations:")
    relevant_interactions = vectorstore.similarity_search(user_input, k=5)
    if relevant_interactions:
        for i, interaction in enumerate(relevant_interactions):
            with st.expander(f"Interaction {i+1}"):
                st.write(f"Content: {interaction.page_content}")
                st.write(f"Metadata: {interaction.metadata}")
    else:
        st.write("No relevant conversations found.")

if st.button("Clear Conversation History"):
    vectorstore.clear()
    st.write("Conversation history cleared.")
