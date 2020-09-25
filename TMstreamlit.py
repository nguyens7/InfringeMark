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
# option = add_selectbox = st.sidebar.selectbox(
#     'How do you want your wordmark to be evaluated?',
#     ('Levenshtein', 'Similarity', 'Phoneme')
#  )

# Add a slider to the sidebar:
# yr_range = add_slider = st.sidebar.slider(
#     'Select a date range',
#     1900, 2020, (1950, 2000)
# )

	def get_ratio(row):
	    name = row['wordmark']
	    return fuzz.token_sort_ratio(name, clean_text)

	menu = ["Levenshtein", "Similarity", "Phoneme"]
	choice = st.sidebar.selectbox("How do you want your trademark to be evaluated", menu)

	if choice == "Levenshtein":
		nlp = spacy.load("en_core_web_lg")
		raw_text = st.text_area("Trademark Search","Enter your trademark here")
		clean_text = str.lower(raw_text)
		tokens = nlp(clean_text)
		if st.button("Uniquness Calculator"):

			# spaCy tokens
			# spacy_streamlit.visualize_tokens(tokens)

			# Import TM data
			df = pd.read_csv("Data.nosync/TM_clean.csv",nrows = 1e6, index_col = False)

			df_matches = df[df.apply(get_ratio, axis=1) > 70]

			df_matches['sim_score']= df.apply(get_ratio, axis=1)
			df_matches = df_matches.sort_values(by='sim_score', ascending=False)

			st.dataframe(df_matches)

	elif choice == "Similarity":
		st.write("That feature hasn't been implemented yet.")

	elif choice == "Phoneme":	
		st.write("That feature hasn't been implemented yet.")

if __name__ == '__main__':
	main()


			# st.text("Similar Trademarks to :", clean_text)


