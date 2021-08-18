import re
import codecs
import matplotlib.pyplot as plt
import numpy as np
import statistics
from math import factorial
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import lxml.html
from lxml.html.clean import Cleaner
import os
import html


'''Attempting Content Extraction with CETR Algorithm
http://hanj.cs.illinois.edu/pdf/www10_tweninger.pdf

1. Get HTML
2. Clean SCRIPT and STYLE tags
3. Split HTML by line
4. For each line, calculate Tag Ratio (Ratio of text to number of tags in the line)
    - Content lines are hypothesized to have a higher Tag Ratio
5.Create a histogram of Tag Ratios 
6. Smooth the histogram
7. Calculate an approximation of the derivatives
8. Use derivatives and smoothed Tag Ratios as 2 features for K-Means Clustering
    - 1 centroid is set to (0,0) 0 tag ratio, and 0 derivative for non-content lines
9. Remove tags and clean from lines classified as content
'''

# Remove style and script tags
def remove_tags(line):
    bad_start = ['<script', '<SCRIPT', '<style']
    bad_end = ['</script>', '</SCRIPT>', '</style>']
    for tag in bad_start:
        if line.startswith(tag):
            return ""
    for tag in bad_end:
        if line.endswith(tag):
            return ""
    remove = re.sub("<!--.*-->", "", line)
    remove = re.sub("<script.*?>.*?</script>", "", remove)
    remove = re.sub("<SCRIPT.*?>.*?</SCRIPT>", "", remove)
    remove = re.sub("<style.*?>.*?</style>", "", remove)
    return remove

# Get the individual lines of the HTML as a list, stripping style and script tags
def get_lines(html):
    # Clean the script and style tags with lxml.html.clean.Cleaner
    temp_fname = 'temp_cleaned.html'
    etree = lxml.html.fromstring(html)
    c = Cleaner()
    c.scripts = True
    c.style = True
    cleaned = c.clean_html(etree)
    # Push new html to a temp file
    with codecs.open(temp_fname, 'w', encoding = 'utf-8') as f:
        f.write(lxml.html.tostring(cleaned, pretty_print = True).decode('utf-8'))
    # Read from temp file and do some more cleaning
    with codecs.open(temp_fname, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    lines = [remove_tags(l) for l in lines]
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l) > 0]
    # Delete temp file
    os.remove(temp_fname)
    return lines

# Get the tag ratio for a line
def get_tr(line):
    tags = re.findall('<.*?>', line)
    no_tags = re.sub('<.*?>', "", line)
    if not tags:
        if not no_tags:
            return 0
        return len(no_tags)
    return len(no_tags) / len(tags)

# Smooth a list of values
def smooth(y, box_pts= 10):
    box = np.ones(box_pts)/ box_pts
    y_smooth = np.convolve(y, box, mode ='same')
    
    #plt.plot(y_smooth, color = 'blue')
    #plt.plot(y, color = 'red')
    return y_smooth

# Get an approximation for the derivatives (Absolute Smoothed Derivatives)
def get_abs_deriv(y, alpha = 3):
    res = []
    for i in range(len(y) - alpha):
        acc = 0
        for j in range(alpha):
            acc += y[i+j]
        res.append((acc / alpha) - y[i])
    
    res = smooth(res)
    res = [abs(v) for v in res]
    #plt.plot(res, color = 'green')
    return res

## 2-Means Clustering
# y is [[attr1], [attr2]]
def cluster(y, iterations = 500):
    # Make all columns same length
    min_len = min([len(attr) for attr in y])
    for i in range(len(y)):
        y[i] = y[i][:min_len]
    
    arr = np.array(y).T
    
    # Initial cluster centroids
    c1 = (0,0)
    c2 = (np.mean(y[0]), np.mean(y[1]))
    #c2 = (1,1)
    
    # Initial clusters
    distances = cdist(arr, [c1, c2], 'euclidean')
    cluster = np.array([np.argmin(i) for i in distances])
    
    prev = cluster
    for i in range(iterations):
        centroids = [c1, arr[cluster == 1].mean(axis = 0)]
        #print(i, centroids)
        
        distances = cdist(arr, centroids, 'euclidean')
        cluster = np.array([np.argmin(i) for i in distances])
        
        if np.array_equal(cluster, prev):
            break
        prev = cluster
    return cluster

# Given the lines and clusters, get the lines classified as content 
def get_content_rows(lines, clusters):
    res = []
    for i in range(len(clusters)):
        if clusters[i] == 1:
            res.append(lines[i])
    return res

# Remove tags from html string
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# remove remaining tags, excess backslashes (\) and \xa0 tag
def clean_string(s):
    res = s
    res = re.sub('<.*?>', ' ', s)
    res = re.sub('\xa0', ' ', s)
    res = re.sub("\\\\", '', s)
    return res

# mai
def cetr_extract(html_file):
    lines = get_lines(html_file)
    trs = [get_tr(l) for l in lines]
    smoothed = smooth(trs)
    #plt.plot(trs, color = 'red', alpha = 0.3)
    #plt.plot(smoothed, color = 'blue', alpha = 0.3)
    der = get_abs_deriv(smoothed)
    content = ''
    # If the content is empty, remove the lines predicted in the previous cluster
    # make new clusters
    while not content:
        clusters = cluster([smoothed, der])
        content_rows = get_content_rows(lines, clusters)
        content = [cleanhtml(html.unescape(row)).strip() for row in content_rows]
        content = clean_string(' '.join(content)).strip()
        content_indices = [i for i,x in enumerate(list(clusters)) if x]
        smoothed = [x for i,x in enumerate(smoothed) if i not in content_indices]
        der = [x for i,x in enumerate(der) if i not in content_indices]
    plt.plot(smoothed, color = 'blue', alpha = 0.3)
    return content

''' Test with a sample file
file = './articles/news374_1.html'
text = './articles/news374_1.txt'
with open(file, 'r', encoding = 'utf-8') as f:
    html_file = f.read()
print(cetr_extract(html_file))
'''