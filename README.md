# Information-Retrieval
경북대학교 정보검색 과제

## 1. DocId, Title, Content Parsing
parse 라이브러리 사용

<img width="800" alt="image" src="https://user-images.githubusercontent.com/54229039/191892089-5db8499f-76c0-481d-b6ca-e8619796c37d.png">
: DocId, Title, Content 분류 Parse 작업 진행

## 2. 형태소 분석
KoNLPy의 Okt 라이브러리 + Komoran 라이브러리 병행 사용

## 3. Tf, Idf 계산
tf*idf를 사용해 weight 측정

## 4. Length Normalization
<img width="800" alt="image" src="https://user-images.githubusercontent.com/54229039/191891947-f2e4ec8d-fb30-4007-8b73-ca620116da97.png">
위의 식을 이용하여 L2 Norm을 구해 Rank 측정 및 정렬

## 5. 결과

1번 doc title인 '지미 카터'가 잘 출력되는 것을 확인

<img width="500" alt="image" src="https://user-images.githubusercontent.com/54229039/191892030-5d0af317-632d-4396-bc0d-966afcce642b.png">

형태소 분석이 어려운 '라부아지에'가 잘 출력되는 것을 확인

<img width="500" alt="image" src="https://user-images.githubusercontent.com/54229039/191892399-c0f67da7-c04e-40de-8467-91e003990524.png">

문장의 경우도 Doc 99(아르테미스)가 상단에 위치함을 확인

<img width="600" alt="image" src="https://user-images.githubusercontent.com/54229039/191892431-cff8b205-8d68-4573-83dd-bf5c606014c7.png">

'프로야구'가 doc에는 '프로 야구'로 존재하지만 제대로 형태소 분석을 해 출력이 됨을 확인

<img width="600" alt="image" src="https://user-images.githubusercontent.com/54229039/191892464-742679c7-fd6f-4809-a580-464c3ed084e8.png">
