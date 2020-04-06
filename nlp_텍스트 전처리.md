```python
#텍스트 정규화

!pip install nltk
```

    Collecting nltk
      Downloading nltk-3.4.5.zip (1.5 MB)
    Requirement already satisfied: six in c:\users\admin\anaconda3\envs\nlp_python\lib\site-packages (from nltk) (1.14.0)
    Building wheels for collected packages: nltk
      Building wheel for nltk (setup.py): started
      Building wheel for nltk (setup.py): finished with status 'done'
      Created wheel for nltk: filename=nltk-3.4.5-py3-none-any.whl size=1449910 sha256=b6437274166876e626abc613c4f0ba2bd7722cd0f9b664f177f6244182635f43
      Stored in directory: c:\users\admin\appdata\local\pip\cache\wheels\48\8b\7f\473521e0c731c6566d631b281f323842bbda9bd819eb9a3ead
    Successfully built nltk
    Installing collected packages: nltk
    Successfully installed nltk-3.4.5
    


```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\admin\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```python
#문장 토큰화(Sentences Tokenized)
from nltk import sent_tokenize
text_sample = "The Matrix is everywhere its all around us, here even in this room. \
               You can see it out your window or on your television. \
               You feel it when you go to work, or go to church or pay your taxes."
sentences = sent_tokenize(text=text_sample)
print(sentences)
print(type(sentences), len(sentences))
```

    ['The Matrix is everywhere its all around us, here even in this room.', 'You can see it out your window or on your television.', 'You feel it when you go to work, or go to church or pay your taxes.']
    <class 'list'> 3
    


```python
#단어 토큰화(Word_Tokenized)
from nltk import word_tokenize

sentence = "The Matrix is everywhere its all around us, here even in this room."
words = word_tokenize(sentence)
print(words)
print(type(words), len(words))
```

    ['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.']
    <class 'list'> 15
    


```python
from nltk import word_tokenize, sent_tokenize

def tokenize_text(text):
    sentences = sent_tokenize(text) # 문장별 분리 토큰
    word_tokens = [word_tokenize(sentence) for sentence in sentences] #문장센텐스(리스트) 를 단어센텐스(리스트) 로 만들기
    return word_tokens

word_tokens = tokenize_text(text_sample)
print(word_tokens)    
print(type(word_tokens), len(word_tokens))
```

    [['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.'], ['You', 'can', 'see', 'it', 'out', 'your', 'window', 'or', 'on', 'your', 'television', '.'], ['You', 'feel', 'it', 'when', 'you', 'go', 'to', 'work', ',', 'or', 'go', 'to', 'church', 'or', 'pay', 'your', 'taxes', '.']]
    <class 'list'> 3
    


```python
# 스톱 워즈 제거(불용어 처리) : is, the, a, will 와 같이 문맥적으로 큰 의미가 없는 단어를 제거
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\admin\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
# NLTK 의 english stopwords 갯수 확인
print('영어 stop words 개수:', len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[:20])
```

    영어 stop words 개수: 179
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his']
    


```python
#스탑워즈를 적용하여 불용어 처리(내가 한거)
from nltk import word_tokenize, sent_tokenize

def tokenize_text(text):
    sentences = sent_tokenize(text) # 문장별 분리 토큰
    word_tokens = [word_tokenize(sentence) for sentence in sentences] #문장센텐스(리스트) 를 단어센텐스(리스트) 로 만들기
    return word_tokens

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

result = []
for token in word_tokens:
    if token not in stopwords:
        result.append(token)

print(result)
print(type(result))

```

    [['The', 'Matrix', 'is', 'everywhere', 'its', 'all', 'around', 'us', ',', 'here', 'even', 'in', 'this', 'room', '.'], ['You', 'can', 'see', 'it', 'out', 'your', 'window', 'or', 'on', 'your', 'television', '.'], ['You', 'feel', 'it', 'when', 'you', 'go', 'to', 'work', ',', 'or', 'go', 'to', 'church', 'or', 'pay', 'your', 'taxes', '.']]
    <class 'list'>
    

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\admin\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
#스탑워즈를 적용하여 불용어 처리(선생님)

import nltk
stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
for sentence in word_tokens:
    filtered_words = []
    for word in sentence:
        word = word.lower()
        if word not in stopwords:
            filtered_words.append(word)
    all_tokens.append(filtered_words)
print(all_tokens)
    
```

    [['matrix', 'everywhere', 'around', 'us', ',', 'even', 'room', '.'], ['see', 'window', 'television', '.'], ['feel', 'go', 'work', ',', 'go', 'church', 'pay', 'taxes', '.']]
    


```python
#문법적 또는 의미적을 변화하는 단어의 원형을 찾는 방법 : Stemming, Lemmatization

#Stemmer(LancasterStemmer)
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('working'), stemmer.stem('works'), stemmer.stem('worked'))
print(stemmer.stem('amusing'), stemmer.stem('amuses'), stemmer.stem('amused'))
print(stemmer.stem('happier'), stemmer.stem('happiest'))
print(stemmer.stem('fancier'), stemmer.stem('fanciest'))

#잘 작동하지 않는다
```


```python
import nltk
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\admin\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
#문법적 또는 의미적을 변화하는 단어의 원형을 찾는 방법 : Lemmatization(WordNetLemmatizer) :정확한 원형 단어 추출을 위해 단어의 품사 입력
from nltk.stem.wordnet import WordNetLemmatizer

lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing', 'v'), lemma.lemmatize('amuses', 'v'), lemma.lemmatize('amused', 'v'))
print(lemma.lemmatize('happier', 'a'), lemma.lemmatize('happiest', 'a'))
print(lemma.lemmatize('fancier', 'a'), lemma.lemmatize('fanciest', 'a'))
```

    amuse amuse amuse
    happy happy
    fancy fancy
    


```python
# ndarray 객체 생성
import numpy as np
dense = np.array([[3,0,1],[0,2,0]])
dense
```




    array([[3, 0, 1],
           [0, 2, 0]])




```python
# 희소행렬 - COO 형식 : 0이 아닌 데이터만 별도의 데이터 배열에 저장하고
# 행과 행의 위치를 별도의 배열로 저장
# 희소 행렬 변환을 위해 scipy sparse 패키지를 이용

from scipy import sparse
data = np.array ([3,1,2,])
row_pos = np.array([0,0,1])
col_pos = np.array([0,2,1])

sparse_coo = sparse.coo_matrix((data,(row_pos,col_pos)))

print(sparse_coo)
# sparse_coo.toarray()
```

      (0, 0)	3
      (0, 2)	1
      (1, 1)	2
    


```python
from scipy import sparse

data2 = np.array([1, 5, 1, 4, 3, 2, 5, 6, 3, 2, 7, 8, 1])

row_pos = np.array([0, 0 ,1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5])
col_pos = np.array([2, 5, 0, 1, 3, 4, 5, 1, 3, 0, 3, 5, 0])

#COO  형식으로 변환
sparse_coo = sparse.coo_matrix((data2, (row_pos, col_pos)))

# 행 위치 배열의 고유한 값들의 시작 위치 인덱스를 배열로 생성 (0 두번 뒤 1이 나옴 그 뒤에는 7번째 열에 2가 나옴 ...)
row_pos_ind = np.array([0,2,7,9,10,12,13])

# CSR 형식으로 변환
sparse_csr = sparse.csr_matrix((data2, col_pos,row_pos_ind))

print(sparse_coo)
print()
print(sparse_coo.toarray())
print()
print(sparse_csr)
print()
print(sparse_csr.toarray())
#print(sparse_csr)
```

      (0, 2)	1
      (0, 5)	5
      (1, 0)	1
      (1, 1)	4
      (1, 3)	3
      (1, 4)	2
      (1, 5)	5
      (2, 1)	6
      (2, 3)	3
      (3, 0)	2
      (4, 3)	7
      (4, 5)	8
      (5, 0)	1
    
    [[0 0 1 0 0 5]
     [1 4 0 3 2 5]
     [0 6 0 3 0 0]
     [2 0 0 0 0 0]
     [0 0 0 7 0 8]
     [1 0 0 0 0 0]]
    
      (0, 2)	1
      (0, 5)	5
      (1, 0)	1
      (1, 1)	4
      (1, 3)	3
      (1, 4)	2
      (1, 5)	5
      (2, 1)	6
      (2, 3)	3
      (3, 0)	2
      (4, 3)	7
      (4, 5)	8
      (5, 0)	1
    
    [[0 0 1 0 0 5]
     [1 4 0 3 2 5]
     [0 6 0 3 0 0]
     [2 0 0 0 0 0]
     [0 0 0 7 0 8]
     [1 0 0 0 0 0]]
    


```python
# 피쳐벡터화(Bag of Words)
# DictVectorizer : 문서에서 단어의 사용 빈도를 나타내는 딕셔너리 정보를 입력받아 BOW 인코딩한 수치 벡터로 변환
# 잘 안쓰임
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'A': 1, 'B' : 2}, {'B': 3, 'C': 1}]
X = v.fit_transform(D)
print(v.feature_names_)
print(v.vocabulary_)
print(X)
print()
print(v.transform({'C': 4, 'D': 3})) #C는 4로 값수정해서 0. 0. 4. 으로나오고 D는 없기 때문에 무시


```

    ['A', 'B', 'C']
    {'A': 0, 'B': 1, 'C': 2}
    [[1. 2. 0.]
     [0. 3. 1.]]
    
    [[0. 0. 4.]]
    


```python
# CountVectorizer (비교적 많이 쓰인다)
# 

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',
]

vect = CountVectorizer()
vect.fit(corpus) # fit() 는 데이터를 모델에 학습시킬 때 사용.
print(vect.get_feature_names())
print(vect.vocabulary_)
print()
print(vect.transform(['This is the second document']).toarray())
print()
print(vect.transform(['Something completely new.']).toarray())
print()
print(vect.transform(corpus).toarray())
# second 의 index= 6  ->  1행 6열에 second 2번값 = 2 로 표시(중요성 강조)
```

    ['and', 'document', 'first', 'is', 'last', 'one', 'second', 'the', 'third', 'this']
    {'this': 9, 'is': 3, 'the': 7, 'first': 2, 'document': 1, 'second': 6, 'and': 0, 'third': 8, 'one': 5, 'last': 4}
    
    [[0 1 0 1 0 0 1 1 0 1]]
    
    [[0 0 0 0 0 0 0 0 0 0]]
    
    [[0 1 1 1 0 0 0 1 0 1]
     [0 1 0 1 0 0 2 1 0 1]
     [1 0 0 0 0 1 0 1 1 0]
     [0 1 1 1 0 0 0 1 0 1]
     [0 1 0 0 1 0 0 1 0 0]]
    


```python
# Stop Words 는 문서에서 단어장을 생성할 때 무시할 수 있는 단어. 보통 영어의 한정형용사(관사)나 접속사 등
# 불용어 처리
vect = CountVectorizer(stop_words=["and","is","the","this"]).fit(corpus)
vect.vocabulary_
```




    {'first': 1, 'document': 0, 'second': 4, 'third': 5, 'one': 3, 'last': 2}




```python
#영어 전체에 대한 불용어 처리
vect = CountVectorizer(stop_words="english").fit(corpus)
vect.vocabulary_
```




    {'document': 0, 'second': 1}




```python
# analyzer, tokenizer, token_pattern 등의 인수로 사용할 토큰 생성기 선택
vect = CountVectorizer(analyzer='char').fit(corpus)
vect.vocabulary_
```




    {'t': 16,
     'h': 8,
     'i': 9,
     's': 15,
     ' ': 0,
     'e': 6,
     'f': 7,
     'r': 14,
     'd': 5,
     'o': 13,
     'c': 4,
     'u': 17,
     'm': 11,
     'n': 12,
     '.': 1,
     'a': 3,
     '?': 2,
     'l': 10}




```python
# n-그램은 단어장 생성에 사용할 토큰의 크기 설정
# 모노그램(1-그램)은 토큰 하나만 단어로 사용하며 바이그램(2-그램)은 두개의 연결된 토큰
vect = CountVectorizer(ngram_range=(1, 2)).fit(corpus)
vect.vocabulary_
```




    {'this': 21,
     'is': 5,
     'the': 14,
     'first': 3,
     'document': 2,
     'this is': 22,
     'is the': 6,
     'the first': 15,
     'first document': 4,
     'second': 11,
     'the second': 17,
     'second second': 13,
     'second document': 12,
     'and': 0,
     'third': 19,
     'one': 10,
     'and the': 1,
     'the third': 18,
     'third one': 20,
     'is this': 7,
     'this the': 23,
     'last': 8,
     'the last': 16,
     'last document': 9}




```python
tf-idf(d,t)=tf(d,t)*idf(t)

tf(d,t): term frequency #특정한 단어의 빈도수
idf(t) : inverse document frequency #특정한 단어가 들어 있는 문서의 수에 반비례하는 수
n : #전체 문서의 수
df(t) : #단어 t 를 가진 문서의 수
idf(d,t)=log(n/(1+df(t)))
```


```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidv = TfidfVectorizer().fit(corpus)
    tfidv.transform(corpus).toarray()
```




    array([[0.        , 0.38947624, 0.55775063, 0.4629834 , 0.        ,
            0.        , 0.        , 0.32941651, 0.        , 0.4629834 ],
           [0.        , 0.24151532, 0.        , 0.28709733, 0.        ,
            0.        , 0.85737594, 0.20427211, 0.        , 0.28709733],
           [0.55666851, 0.        , 0.        , 0.        , 0.        ,
            0.55666851, 0.        , 0.26525553, 0.55666851, 0.        ],
           [0.        , 0.38947624, 0.55775063, 0.4629834 , 0.        ,
            0.        , 0.        , 0.32941651, 0.        , 0.4629834 ],
           [0.        , 0.45333103, 0.        , 0.        , 0.80465933,
            0.        , 0.        , 0.38342448, 0.        , 0.        ]])


