import streamlit as st
import pickle
import string
import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

my_model=pickle.load(open('model.pkl','rb'))
my_vector=pickle.load(open('vectorizer.pkl','rb'))
stop_words=pickle.load(open('stop.pkl','rb'))


def text_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            if i not in stop_words and i not in string.punctuation:
                y.append(ps.stem(i))

    return " ".join(y)

st.title('SMS-SPAM-CLASSIFIER')
st.header('input text you want to test')
text=st.text_input('text',placeholder=None)

if st.button('classify'):
    #data = "Hello there! You will now be amongst the first to hear the details of our special events hosted at TCS. Reply STOP to unsubscribe"
    data = text_preprocess(text)
    vector = my_vector.transform([data]).toarray()
    res=my_model.predict(vector)[0]

    if res == 1:
        st.error('SPAM')
    else:
        st.success('NOT SPAM')

