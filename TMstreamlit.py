# Core Pkgs
import pandas as pd
import streamlit as st
import numpy as np
import PIL

# NLP Pkgs
import jellyfish
import spacy
import spacy_streamlit 
from spacy_streamlit import visualize_textcat
from rapidfuzz import fuzz
from rapidfuzz import process

# Model pkg
import joblib

#Featurizer packges
import unidecode
from fuzzywuzzy import fuzz
import jellyfish
from abydos.distance import (IterativeSubString, BISIM, DiscountedLevenshtein, Prefix, LCSstr, MLIPNS, Strcmp95,
							MRA, Editex, SAPS, FlexMetric, JaroWinkler, HigueraMico, Sift4,
							Eudex, ALINE, PhoneticEditDistance)
from abydos.phonetic import PSHPSoundexFirst, Ainsworth
from abydos.phones import *
import re
from sklearn.preprocessing import MinMaxScaler

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

def sum_ipa(name_a, name_b):
    feat1 = ipa_to_features(pe.encode(name_a))
    feat2 = ipa_to_features(pe.encode(name_b))
    if len(feat1) <= 1:
        score = sum(cmp_features(f1, f2) for f1, f2 in zip(feat1, feat2))/1
    else:    
        score = sum(cmp_features(f1, f2) for f1, f2 in zip(feat1, feat2))/len(feat1)
    return score


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
    
    df['sum_ipa'] = df.apply(lambda row: sum_ipa(row.TM_A,row.TM_B), axis=1)
    
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

# load Model For Gender Prediction
TM_XGB_model = open("Data.nosync/TM_XGboost_classifier.pkl","rb")
TM_clf = joblib.load(TM_XGB_model)

# Prediction
def predict_TM_outcome(data):
	""""A function to predict the similarity of a given trademark"""
	result = TM_clf.predict(data)
	proba_result = TM_clf.predict_proba(data)
	best_pred_result = [p[1] for p in proba_result]
	data['XGB_predict'] = result
	data['XGB_proba'] = best_pred_result
	return data

def main():
	""""A NLP app to identify infringing trademarks using a XGboost model"""

	# Header
	logo = PIL.Image.open('Figs/InfringeMark_logo.png')
	st.image(logo, width=700, output_format='PNG') # logo

	st.write("""
	# An Unofficial USPTO Trademark Search  
	A web application to identify potential infringing trademarks""")

	# Streamlit App
	# nlp = spacy.load("en_core_web_lg")
	raw_text = st.text_area("","Enter your trademark here")
	clean_text = str.lower(raw_text)
	nospace_clean_text = clean_text.replace(' ', '')
	# tokens = nlp(clean_text)
	if st.button("Find Similar Trademarks"):

		if len(clean_text) < 3:
			st.write("Please use a Trademark longer than 3 characters.")

		else:
			st.write("Looking for Trademarks similar to: ", raw_text, ".")

			# spaCy tokens
			# spacy_streamlit.visualize_tokens(tokens)

			# Import TM data
			df = pd.read_csv("https://infringemark.s3.us-east-2.amazonaws.com/TM_clean.csv", index_col = False)
			# df = pd.read_csv("Data.nosync/TM_clean.csv", index_col = False) # Original

			# Filter to make df smaller
			first_char = clean_text[0]
			last_char = clean_text[-6:]
			combo = '^' + first_char + '|' + last_char + '$'
			df_small = df[df['wordmark'].str.contains(combo, na = False)]
			df = df_small

			# Levenshtein fuzzy match
			def get_ratio(row):
				    name = row['wordmark']
				    return fuzz.token_sort_ratio(clean_text, name)

			# Matching
			df_matches = df[df.apply(get_ratio, axis = 1) > 60] 
			df_matches['sim_score'] = df.apply(get_ratio, axis = 1)

			# # Return df
			# st.dataframe(df_matches_sorted)

			# spaCy similarity
			# top_hit = df_matches['wordmark'].iloc[0]
			# nlp_top_hit = nlp(top_hit)
			# spacy_score = nlp_top_hit.similarity(tokens)
			# spacy_score = round(spacy_score, -3)
			# st.write("The similarity of: ",clean_text, "to", top_hit, "is :", spacy_score)

			# Gradient Boost df
			df_GB = df_matches[['wordmark']]
			df_GB['Input'] = clean_text
			df_GB = featurize(df_GB)
			df_GB_features = df_GB.drop(['a', 'b', 'TM_A', 'TM_B'],1)
			df_GB_TM = df_GB[['a', 'b']]
			predict_score = predict_TM_outcome(df_GB_features)
			predict_score = predict_score[['XGB_proba', 'XGB_predict']]
			df_GB = pd.concat([df_GB_TM, predict_score], axis=1, sort=False)
			predict_features = predict_score.drop(['XGB_proba', 'XGB_predict'],1)
			# df_GB = pd.concat([df_GB, df_GB_features], axis=1, sort=False)
			df_GB_result = df_GB.sort_values(by='XGB_proba', ascending=False)
			df_GB_result['XGB_proba'] = round(df_GB_result['XGB_proba'], 2)
			count = len(df_GB[df_GB['XGB_proba'] > 0.65])

			# Return df
			st.dataframe(df_GB_result)

			# InfringeMark recommendation
			if count > 10:
				st.write("InfringeMark recommends to NOT FILE \"",raw_text, "\" for a trademark.\n There are over ", count-1, "similar trademarks." )

			elif count < 10:
				st.write("InfringeMark recommends to FILE \"" ,raw_text, "\" for a trademark.\n There are less than 10 similar trademarks.")

	# Footer
	USPTO_logo = PIL.Image.open('Figs/USPTO_Logo.jpg')
	st.image(USPTO_logo, width=250, output_format='JPG') # logo

	# Disclaimer
	st.write(""" **Disclaimer:** The information provided by this web app does not, and is not intended to, constitute legal advice;
		instead, all information provided by this app is for general informational purposes only.  
		The code behind this app can be accessed on [Github](https://github.com/nguyens7/InfringeMark).""")


if __name__ == '__main__':
	main()


