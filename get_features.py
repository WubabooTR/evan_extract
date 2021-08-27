import lxml.html
from nltk.corpus import stopwords

from collections import defaultdict
import codecs
import re


'''
get_features.py is used to extract labelled features from article HTML to be used as a training dataset
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
STOPWORDS = set([word for word in stopwords.words('english') if "'" not in word])

## Given a HTML of an article, and its related text, 
## Get a list of all the nodes in the HTML and label them with features and content/non-content
##  classification
class GetNodeFeatures():
    def __init__(self, html, text, file = ''):
        self.html = html
        self.root = lxml.html.fromstring(html)
        self.text = text
        self.file = file
        
        self.nodes = self.classify()
        self.features = self.search()
        
    # Iterate through all nodes in the HTML, labelling them appropriately
    def search(self):
        self.features = []
        
        # Store the classification and the tag of the previous node
        prev_node = False
        prev_node_tag = None
        
        for e in self.root.iter():
            text = trim(self.get_element_content(e)).lower()
            features = {}
            features['file'] = self.file
            features['node'] = e
            features['tag'] = e.tag
            features["text_density"] = self.text_density(e)
            features['link_density'] = self.link_density(e)
            # previous node classification
            features['prev_node_class'] = prev_node
            features['prev_node_tag'] = prev_node_tag
            prev_node = self.nodes[e]
            prev_node_tag = e.tag
            
            features['num_children'] = len(e)
            features['text_length'] = len(text.split())
            features['attrib'] = e.attrib
            features['attrib_len'] = len(e.attrib)
            features['stop_words_count'] = sum(map(lambda s: text.count(s), STOPWORDS))
            features['content'] = self.nodes[e]
            self.features.append(features.copy())
        return self.features
            
    # Classify each node as content or non_content 
    # (If the node text appears in content text file)
    def classify(self):
        self.nodes = {}
        for e in self.root.iter():
            # Check to see if both e.text and e.tail are in the text file
            if e.text:
                if e.tail:  
                    self.nodes[e] = (e.text in self.text) and (e.tail in self.text)
                else:
                    self.nodes[e] = e.text in self.text
            elif e.tail:
                self.nodes[e] = e.tail in self.text
            else:
                self.nodes[e] = False
            ''' Does not check for nested content nodes
                - e.g. <tag> content text <child> child text </child> tail text </tag>
            content = self.get_element_content(e)
            self.nodes[e] = (content is not None) and (content in self.text)
            '''
        return self.nodes
    
    # Get the text and tail content of the element
    def get_element_content(self, e):
        content = ''
        if e.text:
            content += e.text
        if e.tail:
            content += e.tail
        return trim(content)
            
    
    # Get length of all text in the elemnet (including children's text)
    def get_all_text_len(self, e):
        if len(e) == 0:
            return len(self.get_element_content(e))
        children_len = 0
        for child in e:
            children_len += self.get_all_text_len(child)
        return len(self.get_element_content(e)) + children_len
    
    # Get a dictionary with counts of all the tags under the root
    def get_tags(self, e):
        tags = defaultdict(int)
        total = 0
        for e in self.root.iter():   
            tags[e.tag] += 1
            total += 1
        return tags, total
    
    # text density is defined as:
    # (total characters in subtree) / (total number of tags in the subtree (=1 if it is 0))
    def text_density(self, e):
        elem_len = self.get_all_text_len(e)
        tags = max(1, self.get_tags(e)[1])
        return elem_len / tags
    
    # link density here will be defined as:
    # (total number of <a> tags) / (total length of text in the block )
    def link_density(self, e):
        LINKS = ['.//link', './/a', './/area']
        link_count = 0
        for tag in LINKS:
            link_count += len(e.findall(tag))
        return link_count / max(1, self.get_all_text_len(e))
        
## TRIM STRINGS
NO_TAG_SPACE = re.compile(r'(?<![p{P}>])\n')
SPACE_TRIMMING = re.compile(r'\s+', flags=re.UNICODE|re.MULTILINE)
def trim(string):
    '''Remove unnecessary spaces within a text string'''
    try:
        # remove newlines that are not related to punctuation or markup + proper trimming
        return SPACE_TRIMMING.sub(r' ', NO_TAG_SPACE.sub(r' ', string)).strip(' \t\n\r\v')
    except TypeError:
        return None

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