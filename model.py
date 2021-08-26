from get_features import GetNodeFeatures, open_file
import lxml.html
from lxml import etree

import os
import tqdm
import datetime
import pandas as pd
import numpy as np
import re 

files = os.listdir('./articles')
files = [f[:-len('.txt')] for f in files if f.endswith('.txt')]

res = []
i = 0
start = datetime.datetime.now()
for file in tqdm.tqdm(files):
    text = './articles/{}.txt'.format(file)
    html = './articles/{}.html'.format(file)
    features = GetNodeFeatures(open_file(html), open_file(text), file)
    res += features.features
    i += 1
    '''
    if i >5:
        break
    '''
    print(i) 

elapsed = (datetime.datetime.now() - start).total_seconds()
df = pd.DataFrame(res)


# Preprocessing 
## - Remove unnecessary columns
## - One Hot Encoding
## - Train Test Split
## - Feature Scaling
def preprocess(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Drop columns 'file', 'node', 'attrib'
    X = X.drop(['file', 'node', 'attrib'], axis = 1)

    # Set types to strings
    X['tag'] = X['tag'].astype(str)    
    X['prev_node_tag'] = X['prev_node_tag'].astype(str)
    X['prev_node_class'] = X['prev_node_class'].astype(int)
    # Set y to 0's and 1's
    y = y.astype(int)
    y = y.values
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 4])], remainder='passthrough', 
                           sparse_threshold = 0)
    X = np.array(ct.fit_transform(X))
    '''
    # One hot encoding
    one_hot_tag = pd.get_dummies(X['tag'])
    one_hot_prev_tag = pd.get_dummies(X['prev_node_tag'])
    one_hot_prev_tag = one_hot_prev_tag.rename(lambda s: "prev_node_{}".format(s), axis = 1)
    X = X.drop(['tag', 'prev_node_tag'], axis = 1)
    X = X.join([one_hot_tag, one_hot_prev_tag])
    '''
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test, ct, sc

# Confusion matrix and accuracy score
def test_model(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = model.predict(X_test)
    side_by_side = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    return cm, acc

def models(X_train, y_train):
    from sklearn.svm import SVC
    classes = {'svc': SVC}
    results = {}
    for key in classes:
        classifier = classes[key]()
        classifier.fit(X_train, y_train)
        results[key] = classifier
    return results

class GetContent():
    def __init__(self, html, model, column_transformer, feature_scaler):
        self.html = html
        self.model = model
        self.root = lxml.html.fromstring(html)
        self.ct = column_transformer
        self.sc = feature_scaler
        
        self.feat = GetNodeFeatures(self.html, '', 'file_name')
        self.features = self.feat.features
        self.nodes = self.feat.nodes
        
        self.classifications = []
        self.content = {}
        self.predict()
        
        self.text = self.get_text(self.features[0]['node'])
        
    def transform(self, node):
        node_df = pd.DataFrame([node])
        node_df = node_df.drop(['file', 'node', 'attrib', 'content'], 1)
        node_df['tag'] = node_df['tag'].astype(str)
        node_df['prev_node_tag'] = node_df['prev_node_tag'].astype(str)
        node_df['prev_node_class'] = node_df['prev_node_class'].astype(int)
        
        node_df = self.sc.transform(self.ct.transform(node_df))
        return node_df
    
    def predict(self):
        
        prev_node_class = 0
        self.classifications = []
        self.temp = []
        for f in self.features:
            f['prev_node_class'] = prev_node_class
            prediction = self.model.predict((self.transform(f)))[0]
            prev_node_class = prediction
            self.classifications.append(prediction)
            self.content[etree.tostring(f['node'])] = prediction
            f['content'] = bool(prediction)
        return self.classifications
    
    # Get the text and tail content of the element
    ### !!!!!!! 
    ## content can be nested, 
    ## e.g. <tag> content text <child> child text </child> tail text </tag>
    ##  - formatting will be messed up!!
    ##      - Recursively: 'content text child text tail text'
    ##      - Iteratively: 'content text tail text child text' 
    ## Recurse through tree and children pulling out text in order
    def get_text(self, node):
        text = ''
        if self.content[etree.tostring(node)] and node.text:
            text += node.text
        for child in node:
            text += ' ' + self.get_text(child)
        if self.content[etree.tostring(node)] and node.tail:
            text += node.tail
        return self.trim(text)
            
    def trim(self, string):
        '''Remove unnecessary spaces within a text string'''
        ## TRIM STRINGS
        NO_TAG_SPACE = re.compile(r'(?<![p{P}>])\n')
        SPACE_TRIMMING = re.compile(r'\s+', flags=re.UNICODE|re.MULTILINE)
        try:
            # remove newlines that are not related to punctuation or markup + proper trimming
            return SPACE_TRIMMING.sub(r' ', NO_TAG_SPACE.sub(r' ', string)).strip(' \t\n\r\v')
        except TypeError:
            return None



X_train, X_test, y_train, y_test, ct, sc = preprocess(df)
start = datetime.datetime.now()
model = models(X_train, y_train)
model_train_time = datetime.datetime.now() - start

cm, acc = test_model(model['svc'], X_test, y_test)