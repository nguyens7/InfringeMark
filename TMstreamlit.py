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
	# InfringeMark app  
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

	def make_clickable(val):
    		return '<a href="{}">{}</a>'.format(val,val)

	menu = ["Levenshtein", "Similarity", "Phoneme"]
	choice = st.sidebar.selectbox("How do you want your trademark to be evaluated", menu)

	if choice == "Levenshtein":
		nlp = spacy.load("en_core_web_lg")
		raw_text = st.text_area("","Enter your trademark here")
		clean_text = str.lower(raw_text)
		tokens = nlp(clean_text)
		if st.button("Find Similar Trademarks"):

			# spaCy tokens
			spacy_streamlit.visualize_tokens(tokens)

			# Import TM data
			df = pd.read_csv("Data.nosync/TM_clean.csv", index_col = False) # nrows = 1e6

			df_matches = df[df.apply(get_ratio, axis=1) > 70]

			df_matches['sim_score'] = df.apply(get_ratio, axis=1)
			df_matches = df_matches.sort_values(by='sim_score', ascending=False)

			# Add urls
			# df_matches['url'] = df_matches['serial_no'].apply(lambda x: f'https://tsdr.uspto.gov/#caseNumber={x}&caseSearchType=US_APPLICATION&caseType=DEFAULT&searchType=statusSearch')

			# df.matches = df_matches.style.format(make_clickable)

			# Return df
			st.dataframe(df_matches)

			if df_matches.shape[0] > 10:
				st.write("InfringeMark recommends to NOT FILE for a trademark.\n There are over ", df_matches.shape[0]-1, "similar trademarks." )

			elif df_matches.shape[0] < 10:
				st.write("InfringeMark recommends to FILE for a trademark.\n There are less than 10 similar trademarks.")


	elif choice == "Similarity":
		st.write("That feature hasn't been implemented yet.")

	elif choice == "Phoneme":	
		st.write("That feature hasn't been implemented yet.")

if __name__ == '__main__':
	main()


			# st.text("Similar Trademarks to :", clean_text)


