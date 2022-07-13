
import pandas as pd
from Helper import make_csv,preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
from deep_translator import GoogleTranslator
import time

try:
    df=pd.read_csv('movie_data.csv',encoding = 'utf-8')
except FileNotFoundError:
    #tic = time.perf_counter()
    make_csv()
    #toc = time.perf_counter()
    #print(f"Создание csv файла заняло {(toc - tic)/60:0.4f} минут")
    #print()
    df=pd.read_csv('movie_data.csv',encoding = 'utf-8')

    

del(df['Unnamed: 0']) 
df['review']=df['review'].apply(preprocessor)
"""
Текст нужно преобразовывать в числовую форму, прежде чем 
их передавать алгоритму МО. Суть метода BOW:
    1. Мы создаём глоссарий уникальных лексем - например, слов - из набора документов
    2. Из каждого документа мы создаём вектор признаков, который содержит счетчики частоты
    появления каждого слова в отдельном документе
"""
count = CountVectorizer(stop_words='english',max_df=.1,max_features=5000)
X=count.fit_transform(df['review'].values)
#tic = time.perf_counter()
lda = LatentDirichletAllocation(n_components=10,random_state=123,learning_method='batch')#10 тем
X_topics = lda.fit_transform(X)
#toc = time.perf_counter()
#print(f"Обучение LDA модели заняло {(toc - tic)/60:0.4f} минут")
#print()

    
#Выведем 7 самых важных слов для каждой из тем
n_topwords = 7
feature_names = count.get_feature_names()
for topic_idx,topic in enumerate(lda.components_):
    print(f'Theme №{topic_idx}')
    print (" ".join([feature_names[i] for i in topic.argsort()\
                     [:- n_topwords-1:-1]]))
        
#Выведем 5 рецензий для одной из тем, выбранных наугад
theme_num = random.randint(0, 9)
theme = X_topics[:,theme_num].argsort()[::-1]
#tic = time.perf_counter()
print(f'Для {theme_num+1} темы найдено 5 отзывов: ')
for iter_idx, movie_idx in enumerate(theme[:5]):
    
    text = df['review'][movie_idx][:500]+"..."
    print(GoogleTranslator(source='auto', target='ru').translate(text))
    print()
#toc = time.perf_counter()
#print(f"Поиск и перевод отзывов на фильм заданной категории занял {toc - tic:0.4f} секунд")
     