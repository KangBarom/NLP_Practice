Data load

glove.6B.50d
glove.840B.300d

위 두 파일은 https://nlp.stanford.edu/projects/glove/ 에서 다운 받을 수 있다.
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
두 파일을 다운 받고 해당 파일의 압축을 ./NLP_Practice/ 에 풀어준다.

그 이후 NLP_Glove_processing.ipnb 순서로 실행하면 된다.
