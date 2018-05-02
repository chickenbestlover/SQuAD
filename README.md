# FusionNet: Fusing via Fully-aware Attention with Application to Machine Comprehension
* 위 코드는 해당 논문( https://openreview.net/forum?id=BJIgi_eCZ )을 구현한 코드 입니다.

# 논문의 내용과 다른점
* 논문에서는 original match, lower match, lemma match, normalized term-frequency 4개의 features를 사용 하였지만 코드에는 original match만 사용 하였습니다.
* 논문에서는 question에서 자주 나오는 단어 top 1000개의 임베딩만 학습을 시키고 나머지는 fix 하였지만, 위 코드에서는 모든 임베딩을 fix 하였습니다.

# 성능표
* F1 = 81.56%
* EM = 72.56%


![scores.jpg](./scores.jpg)
