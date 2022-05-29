import math
import copy
import collections
import numpy as np
from konlpy.tag import *
from konlpy.corpus import kolaw
from parse import *
import operator

DOC_SIZE = 100
# 형태소 분석기
okt = Okt()

query = '미국'

sieve_list = ['Noun', 'Verb', 'Number']
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

        # 명사, 숫자만 저장, 조사 등 의미없는 형태소는 저장 x or morphs[1] == 'Verb' : 로 동사 저장할지?
        if morphs[1] == 'Noun' or morphs[1] == 'Number':
            if docCnt == None:
                voca[morphs[0]] = []
            voca[morphs[0]].append(int(contents['docID']))

# content 검색
for contents in doc:
    pos_list = okt.pos(contents['content'])

    for morphs in pos_list:
        docCnt = voca.get(morphs[0])

        # 명사, 숫자만 저장, 조사 등 의미없는 형태소는 저장 x or morphs[1] == 'Verb' : 로 동사 저장할지?
        if morphs[1] == 'Noun' or morphs[1] == 'Number':
            if docCnt == None:
                voca[morphs[0]] = []
            voca[morphs[0]].append(int(contents['docID']))

#biwords voca에 추가.
# for contents in doc:
#     o_list = okt.pos(contents['content'])
#     sieved_list = [morphs for morphs in o_list if morphs[1] in sieve_list]
#
#     idx = 0
#     biwords_space = ''
#     biwords_no_space = ''
#
#     len_s_list = len(sieved_list)
#     while(idx < len_s_list-1):
#         pre_biwords = sieved_list[idx:idx+2]
#         biwords_space = pre_biwords[0][0] + ' ' + pre_biwords[1][0]
#         biwords_no_space = pre_biwords[0][0] + pre_biwords[1][0]
#
#         #print(biwords_space)
#         #print(biwords_no_space)
#         is_empty = voca.get(biwords_space)
#         is_empty_no_space = voca.get(biwords_no_space)
#
#         if is_empty == None:
#             voca[biwords_space] = []
#
#         if is_empty_no_space == None:
#             voca[biwords_no_space] = []
#
#         voca[biwords_space].append(int(contents['docID']))
#         voca[biwords_no_space].append(int(contents['docID']))
#
#         idx += 1

# tf-idf 계산
tf_matrix = {}

# TF 계산
for term, docList in voca.items():
    tf_matrix[term] = [0] * DOC_SIZE
    for i in range(len(docList)):
        tf_matrix[term][docList[i]-1] += 1

for term, docList in tf_matrix.items():
    for i in range(len(docList)):
        if docList[i] == 0:
            continue
        docList[i] = 1+math.log10(docList[i])

#caculate idf
for term, docList in voca.items():
    voca[term] = copy.deepcopy(set(docList))

    doc_freq = len(voca[term])
    voca[term] = list(voca[term])
    voca[term].sort()
    voca[term].append(math.log10(DOC_SIZE/doc_freq))     #append idf


key_list = list(voca)
query_morphs = okt.pos(query)
sieved_list = [morphs for morphs in query_morphs if morphs[1] in sieve_list]


#key_interests에는 query의 형태소 분해된 결과가 들어감. voca term이랑 일치
key_interests = []
query_vector = [0] * len(voca)
score = [0] * DOC_SIZE

for morphs in query_morphs:
    if morphs[1] in sieve_list:
        key_interests.append(morphs[0])

print('key_interest: ', key_interests)
idx = 0
biwords_space = ''
biwords_no_space = ''

len_s_list = len(sieved_list)

'''while(idx < len_s_list-1):
    pre_biwords = sieved_list[idx:idx+2]
    biwords_space = pre_biwords[0][0] + ' ' + pre_biwords[1][0]
    biwords_no_space = pre_biwords[0][0] + pre_biwords[1][0]

    key_interests.append(biwords_space)
    key_interests.append(biwords_no_space)
    idx += 2
'''
#container = []
# VOCA의 차원과 query_verctor의 차원을 일치 시킨 후 query에 대한 term freq 계산.
for i in range(len(key_interests)):
    query_vector[key_list.index(key_interests[i])] += 1

key_interests = list(set(key_interests))

#log 연산 취함.
for i in range(len(query_vector)):
    if query_vector[i] == 0:
        continue
    query_vector[i] = 1+math.log10(query_vector[i])  #log freq weight

l2_denorm = 0
weight_query = []

#voc_position은 query term의 voca 상의 위치를 나타냄.
#weight_query에는  query term의 tf*idf 값에 대한 normalized(L2_norm)된 결과를 가짐.
#key_interests list에는 query에 나타난 term들을 가짐.
for i in range(len(key_interests)):
    voc_position = key_list.index(key_interests[i])
    tf_qterm = query_vector[voc_position]
    idf_qterm = voca[key_interests[i]][-1]

    weight_query.append({key_interests[i] : float(tf_qterm)*float(idf_qterm)})
    #tf_qterm, idf_qterm 모두 log weight로 계산되어있음.

for i in range(len(weight_query)):
    l2_denorm += pow(weight_query[i][key_interests[i]], 2)

#calculate normalized weight for query term
for i in range(len(weight_query)):
    weight_query[i][key_interests[i]] = weight_query[i][key_interests[i]] / pow(l2_denorm, 1/2)

#doc에 대한 L2_norm 결과값 계산하는 부분
#weight_query : {voca order : weight}
voca_keys = voca.keys()
voca_keys = list(voca_keys)

df_l2_norm = 0

for i in range(DOC_SIZE):
    for j in range(len(voca)):
        df_l2_norm += pow(tf_matrix[voca_keys[j]][i],2)
    df_l2_norm = pow(df_l2_norm, 1/2)
    for k in range(len(voca)):
        tf_matrix[voca_keys[k]][i] = tf_matrix[voca_keys[k]][i] / df_l2_norm

doc_visit_set = []
for i in range(len(weight_query)):
    voc_position = key_list.index(key_interests[i])
    for j in range(len(tf_matrix[key_interests[i]])):
        score[j] = score[j] + (weight_query[i][key_interests[i]] * tf_matrix[key_interests[i]][j])

tuple_score = []
for doc_id, score in enumerate(score):
    tuple_score.append((doc_id+1, score))

# rank 정렬
tuple_score.sort(key = lambda x: x[1], reverse = True)
# print('weight_query: ', weight_query)
print(query, "에 대한 검색 결과 Rank ******")

# rank 상위 5개 출력
for rank in tuple_score[0:5]:
    print(rank[0])