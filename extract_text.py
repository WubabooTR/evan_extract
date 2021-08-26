from lxml import etree
import lxml.html
from collections import defaultdict
import requests
from io import StringIO
import io
import difflib
import re
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
'''
evan's simple text extraction attempt.
Using:
    - text lengths of hardcoded tags.
    - neighbours and parents of nodes.
    - text densities
'''
    

url ='https://www.reuters.com/business/environment/un-sounds-clarion-call-over-irreversible-climate-impacts-by-humans-2021-08-09/'
#url = 'https://www.infowars.com/posts/he-takes-care-of-the-little-girls-leon-blacks-ex-mistress-shares-explosive-new-details-about-best-friend-jeffrey-epstein/'
#url = 'https://leadstories.com/hoax-alert/2021/07/fact-check-study-does-not-prove-that-face-masks-are-dangerous-for-children.html'
file = requests.get(url).text

class Extract():
    def __init__(self, html, text_tags = [], threshold = 0.2):
        self.html = html
        self.root = lxml.html.fromstring(html)
        
        self.text_tags = text_tags
        if not len(self.text_tags): 
            self.text_tags = ['a', 'p', 'strong', 'b', 'em', 'span']
        
        self.threshold = threshold
        self.parent_tags = self.text_tags + ['blockquote']
        
        self.neigh = self.parent_tags + ['figure', 'header', 'footer']
        
        self.tags = {}
        
        self.text_elements, self.text_text = self.search()
        self.clean()
        
    # Get all text elements defined in text_tags
    def search(self):
        text_elements = []
        text_text = []
        self.scores = []
        self.all_checked = []
        for e in self.root.iter(self.text_tags):
            self.all_checked.append(e)
            score = self.classify(e)
            if score:
                text_elements.append(e)
                text_text.append(self.get_element_content(e))
            self.scores.append(score)
            '''
            content = self.get_element_content(e)
    
            # Ignore elements with no children, and little content
            if len(e) == 0 and len(self.get_element_content(e).split()) < 5:
                continue
            
            # If there is at least 7 words in the content, OR
            # There is at least 1 word, some neighbours or parent are valid
            # consider it text
            if (len(content.split()) > 6) or \
            (len(content.split()) > 0 and (self.check_parent(e) or self.neighbours(e) > 1)):
                text_elements.append(e)
                text_text.append(content)
            '''
        self.scores = self.smooth(self.scores)
        text_elements = [self.all_checked[i] for i in range(len(self.scores)) if self.scores[i] > self.threshold]
        text_text = [self.get_element_content(i) for i in text_elements]
        return text_elements, text_text
    
    # Smooth a list of values
    def smooth(self, y, box_pts= 10):
        box = np.ones(box_pts)/ box_pts
        y_smooth = np.convolve(y, box, mode ='same')
        
        plt.plot(y_smooth, color = 'blue')
        plt.plot(y, color = 'red')
        return y_smooth
    
    # Classify node as content or not content
    def classify(self, e):
        content = self.get_element_content(e)
        if len(e) == 0 and len(content.split()) < 5:
            return 0
        elif (self.text_density(e) > 0.3):
            return 1
        elif self.neighbours(e) > 1 or self.check_parent(e):
            return 0.5
        else:
            return 0
        
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
    
    # Check if the parent tags is a specified tag
    def check_parent(self, e):
        par = e.getparent()
        if par is not None and par.tag in self.parent_tags:
            return True
        return False
        
    # Count the number of neighbours with matching tags
    def neighbours(self, e, distance = 3):
        def neighbour_check(e):
            if e is not None and e.tag in self.neigh and len(self.get_element_content(e).split()) > 1:
                return True
            return False
        text_counts = 0
        prev_e = e
        next_e = e
        for i in range(distance):
            prev_e = e.getprevious()
            next_e = e.getnext()
            if neighbour_check(prev_e):
                text_counts += 1
            if neighbour_check(next_e):
                text_counts += 1
        return text_counts
                    
    def clean(self):
        self.clean_text = list(map(lambda s: s.strip(), self.text_text))
        self.clean_text = ''.join(self.clean_text)
    
    # text density is defined as:
    # (total characters in subtree) / (total number of tags in the subtree (=1 if it is 0))
    def text_density(self, e):
        elem_len = self.get_all_text_len(e)
        tags = max(1, self.get_tags(e)[1])
        return elem_len / tags

    
    ''' to add
    - link density
    - more tags
    - find a way to determine which text tags are used
    '''

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

def open_sample(fname):
    with io.open(fname, 'r', encoding = 'utf-8') as f:
        read = f.readlines()
    read = list(map(lambda s: s.strip(), read))
    return ''.join(read)

def check_matches(golden_fname, result):
    golden = open_sample(golden_fname)
    s = difflib.SequenceMatcher(None, golden, result)
    t = difflib.SequenceMatcher(None, result, golden)
    return s.ratio(), t.ratio()