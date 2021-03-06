# Article Extraction
Comparing new article content extraction libraries + methods
  - **Trafilatura:** 
    - https://trafilatura.readthedocs.io/en/latest/index.html
    - Highest Average difflib ratio
  - **Boilerpy3:** 
    - https://github.com/Autonomiq/BoilerPy3
  - **Newspaper3k:**
    - https://newspaper.readthedocs.io/en/latest/
  - **Basic tag filtering:**
    - in extract_text.py
  - **CETR Algorithm:**
    - http://hanj.cs.illinois.edu/pdf/www10_tweninger.pdf
    - in cetr_test.py
  - **model.py**
    - Labelling HTML node features in get_features.py to get training set
    - Testing SVM, Decision Tree, Gradient Boost, Random Forest in model.py
   
### Articles
85 article htmls and texts are located in the articles folder. The texts have been manually extracted from each html. 

### Evaluation:
Evaluation is done by comparing the results of each method's extraction to the golden text file for each article using the diffilb library.
Time elapsed for each method is also recorded.
Results can be found in the results folder:
  **not updated**
  - **res.pkl** contains a pandas dataframe of all method results, the difflib ratio to the golden text, and the time elapsed for each article
  - **df_describe.csv** and **df_describe.pkl** contain the results of ```df.describe()```
  
  **New Results**:
  - **df_new.pkl** contains a DataFrame of all method results, the difflib ratio to the golden text, and the time elapsed for each article
  - **df_new_describe.pkl** and **df_new_describe.csv** contain the resutls of ```df_new.describe()```


