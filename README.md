# 전체 계획

1.pretrained weight 모델을 가져온다.

2.모델에 학습이되지 않은 단어를 분류한다.

3.학습이 되지않은 단어의 오타여부를 체크한다.

4.새로운 단어 학습을 위한 context 크롤링

5.워드임베딩 학습

6.문장임베딩

# 초기작업 -GloVe 데이터를 활용 

1.pretrained weight 모델을 가져온다.

2.모델에 학습이되지 않은 단어를 분류한다.(NLP_processing/NLP_Glove_processing.ipynb 참고)

3.학습이 되지않은 단어의 오타여부를 체크한다.(NLP_processing/NLP_크롤링으로 오타 찾기.ipynb 참고)

4.새로운 단어 학습을 위한 context 크롤링

5.워드임베딩 학습

6.문장임베딩

pretrained weight모델을 활용하기 위해 weight를 가져왔다.

glove.6B.50d

glove.840B.300d

위 두 파일은 https://nlp.stanford.edu/projects/glove/ 에서 다운 받을 수 있다.
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
두 파일을 다운 받고 해당 파일의 압축을 ./NLP_Practice/ 에 풀어준다.

NLP_Glove_processing.ipynb 는 Glove pretrained weight를 가져와 학습된 단어가 없는 것이 무엇인지 분류하는 작업 내용이 들어있다.

하지만 Glove는 online learning으로 확장이 될 수 없다는 것을 알았고 다른 방법을 찾아야 했다.(단어가 없을 때마다 새롭게 학습해야하므로)

# 두번째 작업 skipgram을 활용 

Skip-gram  데이터를 활용하는 방법은 word2vec_git.ipynb 을 참고하면 된다.

Skip gram model 로드하기 위해 아래의 주소에서 weight를 가져온다.

Load Google's pre-trained Word2Vec model.

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 에서 googleNews vector 다운로드
word2vec_git.ipynb를 따르면 학습이 되지 않은 단어를 분류하는 작업까지 마칠 수 있다. 오타여부 체크는 NLP_processing/NLP_크롤링으로 오타 찾기.ipynb 참고하여 변형하자

이 방법 또한 이론적으론 online learning이 가능하지만 gensim api에서 online learning으로 단어 학습하는 작업이 쉽지 않기 때문에 fasttext를 활용하기로 하였다.

# 세번째 작업 fasttext를 활용  
Skip-gram 기반의 모델인 fasttext를 활용한다. 위 모델은 없는 단어가 나올 때 지속적으로 word를 발전 시킬 수 있다는 장점이 있다.  
1.https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md 에서 English: bin, text 두가지를 다운 받는다.  
2.word2vec_git.ipynb를 참고해서 속하지 않은 단어를 찾아낸다. (+ 오타검증은 옵션)  
3.fasttext학습.ipynb를 참고해서 속하지 않은 단어 (oov)를 크롤링한다.  
4.fasttext학습.ipynb를 참고해서 oov학습을 진행한다.   
5.fasttext학습.ipynb를 참고해서 oov가 학습된 모델을 불러 word embedding에 활용한다.


# 네번째 작업 InferSent를 활용하여 unsupervised learning활용
두번째, 세번째 작업을 완료한 후 이 작업을 시행해야한다. 이 작업은 fasttext로 word embedding 되어 학습된 InferSent를 활용한다.  
그리고 우리가 작업한 fasttext weight로 word embedding을 진행한다.  
즉, fasttext로 pretrained된 InferSent 모델을 가져와서 word embedding만 우리가 작업한 fasttext로 대체하는 작업이다.  
1.https://github.com/facebookresearch/InferSent.git 다운을 받는다.   
2.설치 후 InferSent의 root로 들어간 후 아래의 코드를 쳐서 InferSent의 pretrained된 sentece weight를 받자.  
<code> curl -Lo encoder/infersent2.pickle https://dl.fbaipublicfiles.com/infersent/infersent2.pkl</code>  
3.NLP_Practice/NLP_processing/skipgram 안에 데이터들을 1번에서 설치한 InferSent root에 옮겨 놓는다.  
4.NLP_Practice/NLP_processing/infersent사용하기.ipynb jupyter notebook 파일을 참고하여 전체 과정을 이해한다.  
5.NLP_Practice/infersent/unsuperviese_test.py를 1번에서 설치한 InferSent root에 옮겨 놓는다. 
6.unsuperviese_test.py 를 시행한다.  
<code></code>
