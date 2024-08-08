import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load your dataset
@st.cache
def load_data():
    df = pd.read_csv('flipkart_com-ecommerce_sample.csv')  # Update with your path if needed
    return df

df = load_data()

# ML Model setup
@st.cache(allow_output_mutation=True)
def setup_ml_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['product_name'].dropna().tolist(), show_progress_bar=True)
    return model, embeddings

model, embeddings = setup_ml_model()

# Heuristic approach setup
@st.cache(allow_output_mutation=True)
def setup_tfidf_model():
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['product_name'].dropna())
    return tfidf_vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = setup_tfidf_model()

def retrieve_products_ml(query, top_n=10):
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return [(df.iloc[i]['product_name'], similarity_scores[i]) for i in top_indices]

def retrieve_products_tfidf(query, top_n=10):
    query_vec = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return [(df.iloc[i]['product_name'], similarity_scores[i]) for i in top_indices]

st.title('Product Retrieval System')

# Sidebar for selection
st.sidebar.title('Select Retrieval Method')
option = st.sidebar.selectbox('Choose method', ['ML Model (Sentence-BERT)', 'Heuristic Approach (TF-IDF)'])

# Input query from user
query = st.text_input('Enter your search query:')

if query:
    if option == 'ML Model (Sentence-BERT)':
        retrieved_products = retrieve_products_ml(query)
        st.subheader('Results using ML Model (Sentence-BERT)')
    elif option == 'Heuristic Approach (TF-IDF)':
        retrieved_products = retrieve_products_tfidf(query)
        st.subheader('Results using Heuristic Approach (TF-IDF)')

    for product, score in retrieved_products:
        st.write(f"Product: {product} | Similarity: {score:.4f}")
