import streamlit as st
from tensorflow.keras.models import load_model
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Evitar erros de threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# Baixar recursos do NLTK (só precisa uma vez)
@st.cache_resource
def carregar_nltk():
    nltk.download("stopwords")
    nltk.download("rslp")
    return set(stopwords.words("portuguese")), RSLPStemmer()

stopwords_pt, stemmer = carregar_nltk()

st.title("Classificador de textos DOU")

st.text_area("Insira o texto para classificação:", key="input_text")

def limpar_html(texto):
    return BeautifulSoup(texto, "html.parser").get_text()

def normalizar_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if not unicodedata.combining(c)
    )

def limpar_texto(texto):
    """ remove caracteres indesejados e normaliza espaços usando regex"""
    texto = re.sub(r'\d+', ' ', texto) # remove números
    texto = re.sub(r'[^\w\s]', ' ', texto) # remove pontuação
    texto = re.sub(r'\s+', ' ', texto) # normaliza espaço
    return texto.strip().lower()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords_pt and len(t) > 2]


def aplicar_stemming(tokens):
    return [stemmer.stem(t) for t in tokens]

def preprocessar_texto(texto, aplicar_stem=False):
    texto = str(texto)

    texto = limpar_html(texto)
    texto = normalizar_acentos(texto)
    texto = limpar_texto(texto)

    tokens = texto.split()
    tokens = remove_stopwords(tokens)

    if aplicar_stem:
        tokens = aplicar_stemming(tokens)

    return " ".join(tokens)


def get_classificacao(prediction):
    classes = ['Designar', 'Dispensar', 'Exonerar', 'Nomear', 'Tornar sem efeito']  # Exemplo de categorias
    indice = np.argmax(prediction)
    return classes[indice]


@st.cache_resource
def carregar_modelo_e_tokenizer():
    model = load_model("../models/modelo_dou.keras")
    with open("../models/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = f.read()
    tokenizer = tokenizer_from_json(tokenizer_data)
    return model, tokenizer

model, tokenizer = carregar_modelo_e_tokenizer()

if st.button("Classificar") and st.session_state.input_text.strip():
    input_text = st.session_state.input_text
    processed_text = preprocessar_texto(input_text, aplicar_stem=False)
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=500, padding="post", truncating="post")
    prediction = model.predict(padded_sequences)
    # Aqui você pode adicionar a lógica de classificação do texto
    st.write(f"Texto classificado: {prediction}")
    st.write(f"Categoria prevista: {get_classificacao(prediction)}")
    st.success("Classificação concluída!")
else:
    st.info("Digite um texto e clique em 'Classificar'.")

st.markdown("---")
st.markdown("Desenvolvido por Wesin (https://engmoderno.com.br)")

