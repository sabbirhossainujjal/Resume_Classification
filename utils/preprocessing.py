
from tqdm import tqdm
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as word_tokenizer

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


important_punctuations= [p for p in "()-.:;?/_{|}"]
important_punctuations

def remove_stop_words(txt):
    print(f"Total stop words: {len(stop_words)}")
    words = word_tokenizer(txt)
    imp_words = [word for word in words if word not in stop_words and word not in important_punctuations]
    text = ' '.join(imp_words)
    return text

def word_vocab(data):
    vocabulary= {}
    for resume in tqdm(data.values):
        for word in resume.split():
            try:
                vocabulary[word] += 1
            except:
                vocabulary[word] = 1
    sorted_vocab= sorted(vocabulary.items(), key= lambda kv:kv[1], reverse= True)
    print(f"Total unique words in the vocabulary: {len(sorted_vocab)}")
    return vocabulary, sorted_vocab


def char_vocab(resumes):
    chars = {}
    for resume in tqdm(resumes):
        for char in resume:
            try:
                chars[char] += 1
            except:
                chars[char] = 1
    
    sorted_chars= sorted(chars.items(), key= lambda kv: kv[1], reverse= True)
    print(f"Total Unique Characters: {len(chars)}")
    return chars, sorted_chars

def remove_unwanted_chars(txt):
    """This function will make all the input text in lower case to reduce the char vocab size and delete all the unwanted characters and punctuations.
    """
    txt= txt.lower()
    pattern = r"[^a-z\s\d()-.:;?\/^_{|}]" # Discarding all the un-necessary letters keeping english letters and necessary punctuations
    txt= re.sub(pattern, ' ', txt)
    txt= ' '.join(txt.split())
    return txt