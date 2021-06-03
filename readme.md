# Naver Boostcamp AI Tech - P Stage 2

## 개요

### Relation Extraction

문장과 문장 내 개체1의 인덱스, 개체명과 개체2의 인덱스, 개체명이 주어질 때 개체1과 개체2의 관계를 예측하는 문제입니다.

학습 데이터 9,000개와 리더보드 점수를 산출하는데 사용되는 데이터 1,000개로 전체 데이터가 구성되어 있습니다.

두 개체 간의 관계를 나타내는 클래스는 총 42개로 구성되어 있습니다.

## 결과

- Score : 80.0% | Lanking : 43

## 구현

### 파일 구성

- `creators.py` : config에 정의된 모델, 손실함수, 옵티마이저 등을 반환하는 함수와 따로 구현한 모델이나 손실함수, 옵티마이저 등을 정의해놓는 파일입니다.
- `dataset.py` : 데이터셋 클래스를 정의해놓은 파일입니다.
- `inference_*.py` : 테스트 데이터셋에 대한 inference 작업 후에 제출할 결과를 csv 파일로 저장하기 위한 파일입니다.
- `train_*.py` : 모델을 학습하기 위한 파일입니다.
- `utils.py` : Seed를 고정하는 함수와 로그를 출력하는 helper 함수가 정의되어 있는, 유틸리티를 위한 파일입니다.

### 구조

HuggingFace 라이브러리를 이용하여 학습과 추론 과정을 진행하였습니다. 학습에 사용되는 데이터의 수가 9,000개로 매우 적기 때문에 HuggingFace 라이브러리에서 제공하는 pretrained 모델과 토크나이저를 가져와 fine-tuning하는 형태로 학습을 진행했습니다.

argparse를 이용하지 않고 py 파일 내에 config 영역을 따로 만들어서 해당 부분을 수정하면서 실험을 진행했고 이전에 실험한 결과나 인자를 다시 확인할 수 있도록 wandb 라이브러리를 통해 실험 결과와 실험에 사용한 환경을 저장하도록 했습니다.

모델이나 기타 수정사항 등에 대한 전반적인 결과를 빠르게 얻기 위해 train_default.py로 어느정도 성능을 파악한 뒤 최종적으로 쓰일 모델을 학습할 때 k-fold를 사용하여 조금 더 최적화하는 방향으로 시도했습니다.

## 회고

자연어 처리(NLP) 분야에서 대표적인 라이브러리인 HuggingFace를 처음 접하다보니 하나의 모델을 선택해서 점수를 쭉 끌어올려 리더보드에 욕심내는 것보다는 전반적으로 여러 모델을 사용하면서 다른 논문의 모델을 참고해서 모델 아키텍쳐를 조금씩 바꿔보는 등 라이브러리를 확실하게 알아보려고 했던 Competition이었습니다.

여러 모델을 찾아보다가 국내 한 현업에 계신 분께서 개인적으로 한국어 데이터셋을 직접 만들어 학습한 KoELECTRA 모델을 배포하면서 남긴 블로그 기록을 보면서 매우 감명 깊었던 기억이 남았습니다. 자연어 처리 분야가 언어적은 특성을 가지고 있다보니 한국어에 초점이 맞춰진 모델 등이 부족하다고 생각했었는데, 생각에서 그치지 않고 자기가 직접 문제를 해결하고자 모델을 만들고 학습시키고자 실행하시는 모습을 보면서 깊은 감명을 받고 많은 생각을 할 수 있었던 좋은 경험이었습니다.