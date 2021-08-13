from lxml import etree
import lxml.html
from collections import defaultdict
import requests
from io import StringIO
import io
import difflib


url ='https://www.reuters.com/business/environment/un-sounds-clarion-call-over-irreversible-climate-impacts-by-humans-2021-08-09/'
#url = 'https://www.infowars.com/posts/he-takes-care-of-the-little-girls-leon-blacks-ex-mistress-shares-explosive-new-details-about-best-friend-jeffrey-epstein/'
#url = 'https://leadstories.com/hoax-alert/2021/07/fact-check-study-does-not-prove-that-face-masks-are-dangerous-for-children.html'
file = requests.get(url).text

'''
parser = etree.HTMLParser()
tree = etree.parse(StringIO(file), parser)
root = tree.getroot()
'''

text_tags = ['a', 'p', 'strong']

class Extract():
    def __init__(self, html, text_tags = []):
        self.html = html
        self.root = lxml.html.fromstring(html)
        
        self.text_tags = text_tags
        if not len(self.text_tags): 
            self.text_tags = ['a', 'p', 'strong', 'b', 'span', 'em']
        
        self.parent_tags = self.text_tags + ['blockquote']
        
        self.neigh = self.parent_tags + ['figure', 'header', 'footer']
        
        self.tags = {}
        
        self.text_elements, self.text_text = self.search()
        self.clean()
        
    # Get all text elements defined in text_tags
    def search(self):
        text_elements = []
        text_text = []
        for e in self.root.iter(self.text_tags):
            content = self.get_element_content(e)
            content = content.strip()
            ''' # All checked tags
            self.tags[e] = {}
            self.tags[e]['content'] = content
            self.tags[e]['neighbours'] = self.neighbours(e)
            self.tags[e]['parent'] = self.check_parent(e)
            '''
            
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
        return text_elements, text_text
    
    # Get the text and tail content of the element
    def get_element_content(self, e):
        content = ''
        if e.text:
            content += e.text
        if e.tail:
            content += e.tail
        return content
    
    # Get a dictionary with counts of all the tags in the html
    def get_tags(self):
        tags = defaultdict(int)
        for e in self.root.iter():   
            tags[e.tag] += 1
        return tags
    
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
    
    ''' to add
    
    -link densities
    - more tags
    - find a way to determine which text tags are used
    '''
    
'''
# Get all elements with text with at least 2 words and 3 similar neighbours
def search(root, text_tags = text_tags):
    text_elements = []
    text_text = []
    for e in root.iter(text_tags):
        content = ''
        if e.text: 
            content += e.text
        if e.tail: 
            content += e.tail
        if len(content.split()) > 1 and (neighbours(e) > 2 or check_parent(e)):
            text_elements.append(e)
            text_text.append(content)
    return text_elements, text_text
        
# Get a dictionary with counts of all the tags in the html
def get_tags(root):
    tags = defaultdict(int)
    for e in root.iter():
        tags[e.tag] += 1
    return tags

# Calculate the link density of an element
def calc_link_density(element):
    links = 0
    for e in element:
        return

# Check if the parent tag is in the list
def check_parent(element, text_tags):
    par = element.getparent()
    if par is not None and par.tag in text_tags:
        return True
    return False

# Calc number of neighbours with matching tags
def neighbours(element, neigh_distance = 3, text_tags = text_tags):
    def neighbour_check(element):
        if element is not None and element.tag in text_tags:
            return True
    text_counts = 0
    prev_e = element
    next_e = element
    for i in range(neigh_distance):
        prev_e = element.getprevious()
        next_e = element.getnext()
        if neighbour_check(prev_e):
            text_counts += 1
        if neighbour_check(next_e):
            text_counts += 1
    return text_counts
'''


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