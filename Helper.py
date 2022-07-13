import pyprind
import pandas as pd
import os
import numpy as np
import re

"""
Создаём csv-файл из отзыва на фильм и оценки фильму
"""
def make_csv():
    basepath = "D:\ТРИТОН ПРОЕКТЫ\git\\NLPLogicRegression\\aclImdb"
    labels={'pos':1,'neg':0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()
    for s in ('test','train'):
        for l in ('pos','neg'):
            path = os.path.join(basepath,s,l)
            for file in sorted(os.listdir(path)):
                with open (os.path.join(path,file),'r',encoding='utf-8') as infile:
                    txt = infile.read()
                    df =df.append([[txt,labels[l]]],ignore_index=True)
                    pbar.update()
    df.columns=['review','sentiment']
    
    
    np.random.seed(0)#перемешиваем набор данных, так как он отсортирован, что повредит процессу обучения
    df = df.reindex(np.random.permutation(df.index))
   
    df.to_csv('movie_data.csv',encoding = 'utf-8')

def preprocessor(text):
    """
    Отзывы имеют HTML-разметку. 
    Удалим все знаки препинания, отвечающие за неё
    
    """
    text = re.sub('<[^>]*','',text) 
    text = (re.sub('[\W]+',' ',text.lower()))

    return text

