from get_features import GetNodeFeatures, open_file
import lxml.html
from lxml import etree

import os
import tqdm
import datetime
import pandas as pd
import numpy as np
import re 

'''
Evan's attempt at using Machine Learning to classify HTML nodes as content or non-content
'''

'''
1: Create the Training Dataset
    - Iterate through the articles in the articles folder
    - Label each node with features and a classification (content/ not-content)
        Features include:
            - tag
            - text_density ( (total characters in subtree) / (total number of tags in the subtree (=1 if it is 0)))
            - link_density (total number of <a> tags) / (total length of text in the block )
            - classification of previous node (content or non-content)
            - tag of previous node
            - number of children
            - number of words in text and tail
            - number of node attributes
            - count of stop_words
'''
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
    print(i) 

elapsed = (datetime.datetime.now() - start).total_seconds()
df = pd.DataFrame(res)


'''
2:  Preprocessing 
 - Remove unnecessary columns
 - One Hot Encoding
 - Train Test Split
 - Feature Scaling
'''
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
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test, ct, sc

'''
3: Train models
    - Using the training set, train some models 
'''
def models(X_train, y_train):
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    #classes = {'svc': SVC}
    #classes = {'dtc': DecisionTreeClassifier}
    classes = {'rf': RandomForestClassifier, 'gb': GradientBoostingClassifier, 'dt': DecisionTreeClassifier,
               'svc': SVC}
    results = {}
    for key in classes:
        print('training ', key)
        classifier = classes[key]()
        classifier.fit(X_train, y_train)
        results[key] = classifier
    return results

'''
3.5: Get Confusion Matrix and Accuracy Score
'''
def test_model(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = model.predict(X_test)
    side_by_side = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    return cm, acc

'''
4: Make predictions for an HTML file
    - Use the GetContent class to extract the text from the HTML
    - Iterates through the nodes and classifies the nodes as content or non-content
    - Pulls text from all content labelled nodes recursively
    
'''

class GetContent():
    def __init__(self, html, model, column_transformer, feature_scaler):
        self.html = html
        self.model = model
        self.root = lxml.html.fromstring(html)
        self.ct = column_transformer
        self.sc = feature_scaler
        
        # Get features for each node in the HTML
        self.feat = GetNodeFeatures(self.html, '', 'file_name')
        self.features = self.feat.features
        self.nodes = self.feat.nodes
        
        self.classifications = []
        
        # self.content: {str: bool}, bool as 0,1 
        self.content = {}
        self.predict()
        
        # Get the text starting from the root node, taking text from nodes if the node is 
        #   classified as text
        self.text = self.get_text(self.features[0]['node'])
    
    ## Transforms node and its features to a similar form that has been used to preprocess
    ##  the training data
    ##      - gets relevant columns
    ##      - applies one hot encoding
    ##      - applies features scaling
    def transform(self, node):
        node_df = pd.DataFrame([node])
        node_df = node_df.drop(['file', 'node', 'attrib', 'content'], 1)
        node_df['tag'] = node_df['tag'].astype(str)
        node_df['prev_node_tag'] = node_df['prev_node_tag'].astype(str)
        node_df['prev_node_class'] = node_df['prev_node_class'].astype(int)
        
        node_df = self.sc.transform(self.ct.transform(node_df))
        return node_df
    
    ## For each node, make a classification as 'content' or 'non-content'
    def predict(self): 
        prev_node_class = 0
        self.classifications = []
        self.temp = []
        # Iterate through all nodes
        for f in self.features:
            # Set the classification of the previous node
            f['prev_node_class'] = prev_node_class
            # make prediction
            prediction = self.model.predict((self.transform(f)))[0]
            # Update previous node
            prev_node_class = prediction
        
            self.classifications.append(prediction)
            # Add the prediction to a dictionary 
            self.content[etree.tostring(f['node'])] = prediction
            f['content'] = bool(prediction)
        return self.classifications
    
    # Get the text and tail content of the element
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
    
    # Stolen from Trafilatura code
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


'''Run it all'''
X_train, X_test, y_train, y_test, ct, sc = preprocess(df)
start = datetime.datetime.now()
model = models(X_train, y_train)
model_train_time = datetime.datetime.now() - start

cm_gb, acc_gb = test_model(model['gb'], X_test, y_test)
cm_rf, acc_rf = test_model(model['rf'], X_test, y_test)