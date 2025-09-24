import streamlit as st
from tensorflow.keras.models import load_model
import pickle

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
idx2word = tokenizer.index_word
word2idx = tokenizer.word_index

model = load_model("word2vec.h5")

vectors = model.layers[0].trainable_weights[0].numpy()
import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return sorted(list1, reverse=True)[:number_closest]

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        st.write(idx2word[index_word[1]]," -- ",index_word[0])


st.title("Word2Vec Model")

st.write("Please enter a word and the number of associated words.")
requested_word = st.text_input("Insert text")
options = np.arange(1, 16)
requested_number = st.selectbox('Choose a number', options)

if requested_word:
    if requested_word in word2idx:
        if requested_number != None:
            st.subheader(f"Words closest to: **{requested_word}**")
            print_closest(requested_word, requested_number)
    else:
        st.error(f"Word '{requested_word}' not found in vocabulary.")


st.subheader("Word Analogy (e.g. king - man + woman = ?)")

word1 = st.text_input("Word 1 (positive)", key="word1")
word2 = st.text_input("Word 2 (negative)", key="word2")
word3 = st.text_input("Word 3 (positive)", key="word3")
analogy_number = st.selectbox("Number of closest matches", np.arange(1, 11), key="analogy_number")


def display_results(results):
    for similarity, index in results:
        similarity = float(similarity)  # Convert np.float32 → float
        index = int(index)
        word_name = idx2word.get(str(index)) or idx2word.get(index) or f"[{index}]"
        st.write(f"**{word_name}** — similarity: {similarity:.4f}")


if word1 and word2 and word3:
    if word1 in word2idx and word2 in word2idx and word3 in word2idx:
        st.write(f"**{word1} - {word2} + {word3}** results in:")

        try:
            results = compare(
                word2idx[word1],
                word2idx[word2],
                word2idx[word3],
                vectors,
                number_closest=analogy_number
            )

            if results is not None and len(results) > 0:
                display_results(results)
            else:
                st.warning("No results found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("One or more words not found in vocabulary.")
