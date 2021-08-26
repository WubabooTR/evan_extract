from boilerpy3 import extractors
import trafilatura
from newspaper import Article
from newspaper import fulltext
from extract_text import Extract
import cetr_test

from datetime import datetime
import requests
import difflib
import codecs 
import pandas as pd
import os
import numpy as np

## trafila(str, str) -> dict
## Given a url or an html, uses the Trafilatura library to 
##  extract relevant information in dictionary form including:
##      title, author, url, hostname, descriptino, sitename, date, categories, tags, fingerprint
##      id, text, comments, length
## If an error occurs, a dictionary with empty 'text' and 'title' fields is returned
def trafila(url = None, html = None) -> dict:
    if not url and not html:
        return dict([('text', None), ('title', None)])
    try:
        if not html:
            html = trafilatura.fetch_url(url)
        res = trafilatura.bare_extraction(html, include_comments=False, include_tables=False, no_fallback=True)
        if res:
            res["content_length"] = len(res["text"])
        else:
            res = dict([('text', None), ('title', None)])
        return res
    except:
        return dict([('text', None), ('title', None)])
        
## boiler(str, str) -> dict
## Given a url or an html, uses the BoilerPy3 library to extract relevant
## information in dictionary form including:
##      title, text, length
## If an error occurs, a dictionary with empty 'text' and 'title' fields is returned
def boiler(url = None, html = None) -> dict:
    if not url and not html:
        return dict([('text', None), ('title', None)])
    try:
        extractor = extractors.ArticleExtractor()
        d = {}
        if html:
            res = extractor.get_doc(html)
        else:
            res = extractor.get_doc_from_url(url)
        d["title"] = res.title
        d["text"] = res.content
        if res.content:
            d['content_length'] = len(res.content)
        return d
    except:
        return dict([('text', None), ('title', None)])

## Get a dictionary of results returned by Newspaper3k
##      title, text, date, author, length
def newspaper(url = None, html = None):
    if not url and not html:
        return dict([('text', None), ('title', None)])
    try:
        if url:
            article = Article(url)
            article.download()
        else:
            article = Article(url = '')
            article.set_html(html)
        article.parse()
        d = {}
        d["title"] = article.title
        d["text"] = article.text
        if html:
            d["text"] = fulltext(html)
        d["date"] = article.publish_date
        d["author"] = article.authors
        if d["text"]:
            d["content_length"] = len(d["text"])
        return d
    except:
        return dict([('text', None), ('title', None)])

# Evan's
def evan(url = None, html= None, threshold = 0.2):
    if not html:
        return dict([('text', None), ('title', None)])
    try:
        e = Extract(html, threshold = threshold)
        d = {}
        d["title"] = None
        d["text"] = e.clean_text
        return d
    except:
        return dict([('text', None), ('title', None)])

# CETR
def cetr(url = None, html = None):
    if not html:
        return dict([('text', None), ('title', None)])
    try:
        d = {}
        content = cetr_test.cetr_extract(html)
        d["title"] = None
        d["text"] = content
        return d
    except:
        return dict([('text', None), ('title', None)])

    
# open_file(str) -> str
# Given a string path, opens the relevant file and return its contents
# If the file does not exist, returns None
def open_file(file):
    try:
        f = codecs.open(file, 'r', encoding = 'utf-8')
        res = f.read()
        f.close()
        return res
    except:
        return None

## matches(str, str) -> (int, float)
## Uses the difflib library to find the similarity between two strings
## Returns a tuple of the the total size of matched substrings and the difflib ratio
def matches(s1, s2):
    if (not s1) or (not s2) or (not isinstance(s1, str)) or (not isinstance(s2, str)):
        return 0, 0
    d = difflib.SequenceMatcher(difflib.IS_CHARACTER_JUNK, s1, s2)
    e = difflib.SequenceMatcher(difflib.IS_CHARACTER_JUNK, s2, s1)
    matches = d.get_matching_blocks()
    ratio = max(d.ratio(), e.ratio())
    total = 0
    for match in matches:
        total += match.size
    return total, round(ratio, 5)

## Compare the results of the libraries, getting elapsed time, and similarity to 
## true values in html article
def compare(html, true_res):
    libs = {'trafilatura': trafila, 'boilerpy3': boiler, 
            'newspaper3k': newspaper, 'evan': evan, 'cetr': cetr}
    res = {}
    for lib in libs:
        start_time = datetime.now()
        val = libs[lib](url = None, html = html)
        elapsed = (datetime.now() - start_time).total_seconds()
        res[lib + '_text'] = val['text']
        res[lib + '_diff'] = matches(val["text"], true_res)[1]
        res[lib + '_time'] = elapsed
        
        text_len = 0
        if val['text']:
            text_len = len(val['text'])
        res[lib + '_lendiff'] = text_len - len(true_res)
        
    return res

res = []
files = os.listdir('./articles')
files = [f[:-len('.txt')] for f in files if f.endswith('.txt')]

j = 0
for file in files:
    j += 1
    text = './articles/{}.txt'.format(file)
    html = './articles/{}.html'.format(file)
    row = compare(open_file(html), open_file(text))
    
    row['file'] = file
    
    '''test cetr'''
    #row = cetr(html = open_file(html))
    #row['ratio'] = matches(row['text'], open_file(text))[1]
    
    
    '''test evan_extract'''
    #row = {}
    #for i in np.arange(0.05, 0.2, 0.02):
    #    row['text_' + str(i)[:4]] = evan(html = open_file(html), threshold = i)['text']
    #    row['ratio_' + str(i)[:4]] = matches(row['text_' + str(i)[:4]], open_file(text))[1]
    
    res.append(row)

df = pd.DataFrame(res)


