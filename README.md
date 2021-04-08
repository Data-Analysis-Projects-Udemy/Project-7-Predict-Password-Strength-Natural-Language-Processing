# Project7 Predict Password Strength Natural Language Processing
1. [Importar librerías y cargar datos](#schema1)
2. [Comprobar los nulos](#schema2)
3. [Tenerlos datos en una tupla](#schema3)
4. [Crear una función personalizada para dividir la entrada en caracteres de la lista](#schema4)
5. [Importar vectorizador TF-IDF para convertir datos de cadena en datos numéricos](#schema5)
6. [Aplicar LogisticRegression](#schema6)
7. [Documentación](#schema7)


<hr>

<a name="schema1"></a>

# 1 Importar librerías y cargar datos
~~~python
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv("./data/data.csv", error_bad_lines=False)
~~~
![img](./images/001.png)

~~~python
data['strength'].unique()
array([1, 2, 0])
~~~
<hr>

<a name="schema2"></a>

# 2 Comprobar los nulos
~~~python
data.isna().sum()
password    1
strength    0
dtype: int64


data.dropna(inplace = True)
~~~

<hr>

<a name="schema3"></a>

# 3. Tenerlos datos en una tupla
~~~python
password_tuple = np.array(data)
password_tuple
array([['kzde5577', 1],
       ['kino3434', 1],
       ['visi7k1yr', 1],
       ...,
       ['184520socram', 1],
       ['marken22a', 1],
       ['fxx4pw4g', 1]], dtype=object)
       
       
random.shuffle(password_tuple)

x = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]
~~~

<hr>

<a name="schema4"></a>

# 4. crear una función personalizada para dividir la entrada en caracteres de la lista
~~~python
def word_divide_char(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character
word_divide_char('kzde5577')
['k', 'z', 'd', 'e', '5', '5', '7', '7']
~~~
<hr>

<a name="schema5"></a>

# 5. Importar vectorizador TF-IDF para convertir datos de cadena en datos numéricos

~~~python
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
X=vectorizer.fit_transform(x)
X=vectorizer.fit_transform(x)

X.shape
(669639, 132)



first_document_vector=X[0]
first_document_vector
<1x132 sparse matrix of type '<class 'numpy.float64'>'
	with 6 stored elements in Compressed Sparse Row format>


df=pd.DataFrame(first_document_vector.T.todense(),index=vectorizer.get_feature_names(),columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'],ascending=False)
~~~
![img](./images/002.png)



<hr>

<a name="schema6"></a>

# 6. Aplicar LogisticRegression
dividir datos en entrenar y probar

     entrenar ---> Para aprender la relación dentro de los datos,

     prueba -> Para hacer predicciones, y estos datos de prueba no serán vistos 

     para mi modelo


~~~python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
clf=LogisticRegression(random_state=0,solver='liblinear')
clf.fit(X_train,y_train)
~~~
### Hacer la predicción
~~~python
dt=np.array(['%@123abcd'])
pred=vectorizer.transform(dt)
clf.predict(pred)
array([2])
~~~
####  Comprobar la precisión usando confusion_matrix,accuracy_score
~~~python
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
[[ 3476 14561    30]
 [ 2234 94783  2221]
 [   70  5948 10605]]
0.8128546681799176
~~~
####  Crear un reporte para el modelo 
~~~python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
              precision    recall  f1-score   support

           0       0.60      0.19      0.29     18067
           1       0.82      0.96      0.88     99238
           2       0.82      0.64      0.72     16623

    accuracy                           0.81    133928
   macro avg       0.75      0.60      0.63    133928
weighted avg       0.79      0.81      0.78    133928
~~~




<hr>

<a name="schema7"></a>

# 7. Documentación


https://unipython.com/como-preparar-datos-de-texto-con-scikit-learn/