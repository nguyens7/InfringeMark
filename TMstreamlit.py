


# Core Pkgs
import pandas as pd
import streamlit as st


# NLP Pkgs
import spacy
import spacy_streamlit 
from spacy_streamlit import visualize_textcat
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



def main():
	""""A NLP app to identify infringing trademarks using spaCy"""

st.write("""
	# Infringemark app  
	A web application to identify potential infringing trademarks""")


# Add a selectbox to the sidebar:
option = add_selectbox = st.sidebar.selectbox(
    'How do you want your wordmark to be evaluated?',
    ('Similarity', 'Levenshtein', 'Phoneme')
 )

# Add a slider to the sidebar:
yr_range = add_slider = st.sidebar.slider(
    'Select a date range',
    1900, 2020, (1950, 2000)
)

'You selected: ', option

'Looking for trademarks within: ', yr_range

nlp = spacy.load("en_core_web_lg")
raw_text = st.text_area("Trademark Search","Trademark that you want to check")
clean_text = str.lower(raw_text)
tokens = nlp(clean_text)

if st.button("Tokenize"):
	spacy_streamlit.visualize_tokens(tokens)

# for token1 in tokens:
#     for token2 in tokens:
#         visualize_textcat(token1.similarity(token2))


# For a trained classifier
# elif st.button("Classify"):
# 	spacy_streamlit.visualize_textcat(tokens)

# Import pandas df of TMs
	df = pd.read_csv("Data.nosync/TM_clean.csv",nrows = 10000, index_col = False)

	possibilities = process.extract(clean_text, df, limit=10, scorer=fuzz.token_sort_ratio)	

	# st.write(possibilities)
	# df2 = df.head(200)

	st.dataframe(possibilities)









