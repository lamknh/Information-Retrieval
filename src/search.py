import math
import copy
import collections
import numpy as np
from konlpy.tag import *
from konlpy.corpus import kolaw
# from konlpy.tag import Komoran
from PyKomoran import *
from parse import *
import operator

DOC_SIZE = 100
# 형태소 분석기
okt = Okt()
komoran = Komoran("STABLE")
komoran.set_user_dic('.')

query = '마굴리스'

morphList = ['Noun', 'Verb', 'Number']
voca = {} # 단어 당 나온 docID
doc = [] # docID, title, content 저장
f = open('/Users/nahyeongkim/Information-Retrieval/src/corpus.txt', 'r')
i = 0
doc_num = -1

with open('src/corpus.txt', 'r', encoding='utf-8') as f:
    # 한 줄씩 읽기
    line = f.readlines()

    # xml 파싱
    for l in line:
        if l == '\n':
            continue
        elif l[0:7] == '<title>':
            doc_contents = parse('<title>{docID}. {title}</title>\n', l)
            # parse 결과 딕셔너리 doc 리스트에 저장
            if doc_contents is not None:
                doc.append(doc_contents.named)
                doc_num += 1
        else:
            # doc 내용 저장
            doc[doc_num]['content'] = l

# title 검색
for contents in doc:
    # 텍스트에 품사 정보 붙여 반환
    title_list = okt.pos(contents['title'])

    for morphs in title_list:
        # 형태소 morphs[0] : 형태소, morphs[1] : 형태소 분류, docCnt : 나온 doc ID
        docCnt = voca.get(morphs[0])

        # 명사, 숫자, 동사만 저장, 조사 등 의미없는 형태소는 저장 x
        if morphs[1] in morphList:
            if docCnt == None:
                voca[morphs[0]] = []
            voca[morphs[0]].append(int(contents['docID']))

# content 검색
for contents in doc:
    pos_list = okt.pos(contents['content'])
    k_pos_list = komoran.nouns(contents['content'])

    for morphs in pos_list:
        docCnt = voca.get(morphs[0])

        # 명사, 숫자, 동사만 저장, 조사 등 의미없는 형태소는 저장 x
        if morphs[1] in morphList:
            if docCnt == None:
                voca[morphs[0]] = []
            voca[morphs[0]].append(int(contents['docID']))

    for morphs in k_pos_list:
        docCnt = voca.get(morphs)

        if docCnt == None:
            voca[morphs] = []
        voca[morphs].append(int(contents['docID']))

# print(voca)
#biwords voca에 추가할지?

# tf-idf 계산
tf_matrix = {}

# tf 계산 : term이 doc에 몇 번 나타나는가? 횟수에 비례하지 않고 log 사용
for term, docList in voca.items():
    # 2차원 배열 사용
    tf_matrix[term] = [0] * DOC_SIZE
    for i in range(len(docList)):
        # docID에서 term이 나오는 수 세기 (tf)
        tf_matrix[term][docList[i]-1] += 1

# tf weight 계산
for term, docList in tf_matrix.items():
    for i in range(len(docList)):
        if docList[i] != 0:
            # Wt,d = 1 + log10 tft,d
            docList[i] = 1 + math.log10(docList[i])

# idf 계산 : rare term more informative
for term, docList in voca.items():
    # tf 세기
    docFreq = len(docList)
    # idft = log10 (N/dft)
    voca[term].append(math.log10(DOC_SIZE/docFreq))

key_list = list(voca)

# 쿼리 형태소 분석
query_morphs = okt.pos(query)
k_query_morphs = komoran.nouns(query)
queryList = [morphs for morphs in query_morphs if morphs[1] in morphList]
queryList.append(k_query_morphs)

query_key = [] # query의 형태소 분해된 결과
query_vector = [0] * len(voca)
score = [0] * DOC_SIZE

for morphs in query_morphs:
    if morphs[1] in morphList:
        query_key.append(morphs[0])

for morphs in k_query_morphs:
    query_key.append(morphs)

# 중복 제거
query_key = set(query_key)
query_key = list(query_key)

# print(query_key)

# query tf 계산.
for i in range(len(query_key)):
    if query_key[i] in key_list:
        query_vector[key_list.index(query_key[i])] += 1

# tf weight 계산
for i in range(len(query_vector)):
    if query_vector[i] != 0:
        # Wt,d = 1 + log10 tft,d
        query_vector[i] = 1 + math.log10(query_vector[i])

# tf * idf 계산
weight_query = []

for i in range(len(query_key)):
    # query term 위치
    if query_key[i] in key_list:
        voc_position = key_list.index(query_key[i])
        # query term의 tf, idf 값 가져오기
        tf_qterm = query_vector[voc_position]
        idf_qterm = voca[query_key[i]][-1]

        # tf * idf 계산
        weight_query.append({query_key[i]: float(tf_qterm) * float(idf_qterm)})
    else :
        weight_query.append({query_key[i]: 0})

# Length Normalization

# query L2 norm 구하기
L2_temp = 0
for i in range(len(weight_query)):
    # q^2
    if query_key[i] in key_list:
        L2_temp += pow(weight_query[i][query_key[i]], 2)

# 루트 q^2
query_L2_norm = pow(L2_temp, 1/2)

for i in range(len(weight_query)):
    if query_key[i] in key_list:
        weight_query[i][query_key[i]] = weight_query[i][query_key[i]] / query_L2_norm

# doc L2 norm 구하기
voca_terms = list(voca.keys())

doc_L2_norm = 0

for i in range(DOC_SIZE):
    L2_temp = 0
    for j in range(len(voca)):
        L2_temp += pow(tf_matrix[voca_terms[j]][i], 2)

    doc_L2_norm = pow(L2_temp, 1/2)

    for k in range(len(voca)):
        tf_matrix[voca_terms[k]][i] = tf_matrix[voca_terms[k]][i] / doc_L2_norm

# cos 값 구하기
for i in range(len(weight_query)):
    if query_key[i] in key_list:
        for j in range(len(tf_matrix[query_key[i]])):
            score[j] += (weight_query[i][query_key[i]] * tf_matrix[query_key[i]][j])

# 계산 결과 저장
scoreSet = []

for doc_id, score in enumerate(score):
    scoreSet.append((doc_id+1, score))

# rank 정렬
scoreSet.sort(key = lambda x: x[1], reverse = True)
print("'", query, "'에 대한 검색 결과 Rank ******")

# rank 상위 5개 출력
for rank in scoreSet[0:5]:
    print(rank)
    # print(rank[0])