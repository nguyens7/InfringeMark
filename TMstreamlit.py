# Core Pkgs
import pandas as pd
import streamlit as st

# NLP Pkgs
import jellyfish
import spacy
import spacy_streamlit 
from spacy_streamlit import visualize_textcat

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
# from rapidfuzz import fuzz
# from rapidfuzz import process

# Model pkg
import joblib

#Featurizer packges

import unidecode
from fuzzywuzzy import fuzz
import jellyfish

from abydos.distance import (IterativeSubString, BISIM, DiscountedLevenshtein, Prefix, LCSstr, MLIPNS, Strcmp95,
	MRA, Editex, SAPS, FlexMetric, JaroWinkler, HigueraMico, Sift4, Eudex, ALINE, PhoneticEditDistance)

from abydos.phonetic import PSHPSoundexFirst, Ainsworth

from abydos.phones import *
import re
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


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

    # load Model For Gender Prediction
	TM_GBC_model = open("Data.nosync/TM_Gradient_boost_base_model.pkl","rb")
	TM_clf = joblib.load(TM_GBC_model)

	# Prediction
	def predict_TM_outcome(data):
		result = TM_clf.predict(data)
		return result

	# Featurizer
	
	pshp_soundex_first = PSHPSoundexFirst()
	pe = Ainsworth()	
	iss = IterativeSubString()
	bisim = BISIM()
	dlev = DiscountedLevenshtein()
	prefix = Prefix()
	lcs = LCSstr()
	mlipns = MLIPNS()
	strcmp95 = Strcmp95()
	mra = MRA()
	editex = Editex()
	saps = SAPS()
	flexmetric = FlexMetric()
	jaro = JaroWinkler(mode='Jaro')
	higuera_mico = HigueraMico()
	sift4 = Sift4()
	eudex = Eudex()
	aline = ALINE()
	phonetic_edit = PhoneticEditDistance()
	algos = [iss, bisim, dlev, prefix, lcs, mlipns, strcmp95, mra, editex, saps, flexmetric, jaro, higuera_mico, sift4, eudex,
         aline, phonetic_edit]

	algo_names = ['iterativesubstring', 'bisim', 'discountedlevenshtein', 'prefix', 'lcsstr', 'mlipns', 'strcmp95', 'mra',
              'editex', 'saps', 'flexmetric', 'jaro', 'higueramico', 'sift4', 'eudex', 'aline',
              'phoneticeditdistance']


	def featurize(df):
	    if len(df.columns)==3:
	        df.columns=['a', 'b', 'target']
	    elif len(df.columns)==2:
	        df.columns=['a', 'b']
	    else:
	        df = df.rename(columns={df.columns[0]: 'a', df.columns[1]: 'b' })
	        
	    df['TM_A'] = df.apply(lambda row: re.sub(
	        '[^a-zA-Z]+', '', unidecode.unidecode(row['a']).lower()), axis=1)
	    df['TM_B'] = df.apply(lambda row: re.sub(
	        '[^a-zA-Z]+', '', unidecode.unidecode(row['b']).lower()), axis=1)
	    

	    df['partial'] = df.apply(lambda row: fuzz.partial_ratio(row.TM_A,row.TM_B), axis=1)
	    df['tkn_sort'] = df.apply(lambda row: fuzz.token_sort_ratio(row.TM_A,row.TM_B), axis=1)
	    df['tkn_set'] = df.apply(lambda row: fuzz.token_set_ratio(row.TM_A,row.TM_B), axis=1)
	    
		# df['sum_ipa'] = df.apply(lambda row: sum_ipa(row.TM_A,row.TM_B), axis=1)
	    
	    # Jellyfish levenshtein
	    df['levenshtein']= df.apply(lambda row: jellyfish.levenshtein_distance(row.TM_A,row.TM_B), axis=1)
	    # Scale Levenshtein column
	    scaler = MinMaxScaler()
	    df['levenshtein'] = scaler.fit_transform(df['levenshtein'].values.reshape(-1,1))

	    # Jellyfish phoneme
	    df['metaphone'] = df.apply(
	        lambda row: 1 if jellyfish.metaphone(row.TM_A)==jellyfish.metaphone(row.TM_B) else 0, axis=1)
	    df['nysiis'] = df.apply(
	        lambda row: 1 if jellyfish.nysiis(row.TM_A)==jellyfish.nysiis(row.TM_B) else 0, axis=1)
	    df['mtch_rtng_cdx'] = df.apply(
	        lambda row: 1 if jellyfish.match_rating_codex(row.TM_A)==jellyfish.match_rating_codex(row.TM_B) else 0, axis=1)
	    
	    df['pshp_soundex_first'] = df.apply(
	        lambda row: 1 if pshp_soundex_first.encode(row.TM_A)==pshp_soundex_first.encode(row.TM_B) else 0, axis=1)
	    
	    for i, algo in enumerate(algos):
	            df[algo_names[i]] = df.apply(lambda row: algo.sim(row.TM_A, row.TM_B), axis=1)

	    
	    return df


	# Streamlit App

	menu = ["Levenshtein", "Similarity", "Phoneme"]
	choice = st.sidebar.selectbox("How do you want your trademark to be evaluated", menu)

	if choice == "Levenshtein":
		nlp = spacy.load("en_core_web_lg")
		raw_text = st.text_area("","Enter your trademark here")
		clean_text = str.lower(raw_text)
		nospace_clean_text = clean_text.replace(' ', '')
		tokens = nlp(clean_text)
		if st.button("Find Similar Trademarks"):

			# spaCy tokens
			spacy_streamlit.visualize_tokens(tokens)

			# Import TM data
			df = pd.read_csv("Data.nosync/TM_clean_soundex.csv", index_col = False) # nrows = 1e6
			df_matches = df[df.apply(get_ratio, axis = 1) > 70]
			df_matches['sim_score'] = df.apply(get_ratio, axis = 1)
			df_matches = df_matches.sort_values(by = 'sim_score', ascending = False)

			# Add urls
			# df_matches['url'] = df_matches['serial_no'].apply(lambda x: f'https://tsdr.uspto.gov/#caseNumber={x}&caseSearchType=US_APPLICATION&caseType=DEFAULT&searchType=statusSearch')

			# df.matches = df_matches.style.format(make_clickable)

			# Return df
			st.dataframe(df_matches)

			# spaCy similarity
			top_hit = df_matches['wordmark'].iloc[0]
			nlp_top_hit = nlp(top_hit)
			spacy_score = nlp_top_hit.similarity(tokens)
			# spacy_score = round(spacy_score, -3)
			st.write("The similarity of: ",clean_text, "to", top_hit, "is :", spacy_score)

			if df_matches.shape[0] > 10:
				st.write("InfringeMark recommends to NOT FILE for a trademark.\n There are over ", df_matches.shape[0]-1, "similar trademarks." )

			elif df_matches.shape[0] < 10:
				st.write("InfringeMark recommends to FILE for a trademark.\n There are less than 10 similar trademarks.")

			# Gradient Boost df
			df_GB = df_matches[['wordmark']]
			df_GB['Input'] = clean_text
			df_GB = featurize(df_GB)

			# Return df
			st.dataframe(df_GB)


	elif choice == "Similarity":
		st.write("That feature hasn't been implemented yet.")

	elif choice == "Phoneme":	
		st.write("That feature hasn't been implemented yet.")

if __name__ == '__main__':
	main()


			# st.text("Similar Trademarks to :", clean_text)


