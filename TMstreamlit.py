

import streamlit as st
import pandas as pd
import altair as alt

st.write("""
	# Infringemark app  
	A web application to identify potential infringing trademarks""")


# Add a wordmark to the sidebar:
txt_input = st.sidebar.text_input('Input your wordmark here:')

# Add a selectbox to the sidebar:
option = add_selectbox = st.sidebar.selectbox(
    'How do you want your wordmark to be evaluated?',
    ('Similarity', 'Levehnstein', 'Phenome')
 )

# Add a slider to the sidebar:
yr_range = add_slider = st.sidebar.slider(
    'Select a date range',
    1900, 2020, (1950, 2000)
)

'You selected: ', option

'Looking for trademarks within: ', yr_range

'The wordmark that you want to look for is: ', txt_input