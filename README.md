# BUS-INT

pip install -U jupyterlab ipywidgets  
pip install -U pandas matplotlib seaborn scipy scikit-learn joblib lxml beautifulsoup4 pillow sqlalchemy openpyxl xlrd statsmodels tabulate pandas-datareader pyarrow  
pip install -U plotly  
pip install -U jupysql duckdb duckdb-engine  
pip install -U graphviz  
pip install -U lightgbm   
Pip install scikit-learn  
pip install -U yellowbrick  
pip cache purge

from import  
import pandas as pd  
import seaborn as sns  
import numpy as np  
import matplotlib.pyplot as plt  
import ipywidgets as widgets  
from ipywidgets import interact, interactive, fixed, interact_manual  
import warnings  
warnings.filterwarnings("ignore")  

%matplotlib inline  
%config InlineBackend.figure_format = 'retina'  

print(f'pandas  version: {pd.__version__}')  
print(f'numpy   version: {np.__version__}')  
print(f'seaborn version: {sns.__version__}')  
print(pd.Timestamp.now())  


form… import… sklearn: LinearRegression  
import sklearn  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics  
from sklearn.cluster import KMeans  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import (confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    f1_score,    


print(f'sklearn version: {sklearn.__version__}')  
model = LinearRegression()  


import : Statsmodels  
import statsmodels.api as sm  
import statsmodels.formula.api as smf
