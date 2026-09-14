# 구현 근거와 차이

이 구현은 기존 멀티모달 SAE 연구의 모델·통계 정의를 참고하되, 공식 COCO2017 분할과 객체 제거 기반 진단을 독립된 실험으로 구성했다. 기존 논문의 전체 학습과 결과를 그대로 재현했다고 주장하지 않는다.

SAE는 `/Users/jihoonkwon/Desktop/Projects-old/lvlm_hallucination/src/models/modeling_sae.py`의 `TopKSAE`를 참고했다. 검토한 저장소 커밋은 `0c6b74406e78b15e7101d41a50134bb4bd4c935e`다. 입력에서 디코더 편향을 빼고, 선형 인코더와 ReLU를 적용한 뒤, 가장 큰 활성값만 남긴다. 인코더 가중치를 복사해 별도의 디코더를 초기화하고 자료 묶음의 분산으로 복원 오차를 나눈다. 매개변수 갱신 뒤 디코더 특징 벡터 길이를 정규화한다.

실제 학습 연결은 `/Users/jihoonkwon/Desktop/Projects-old/lvlm_hallucination/scripts/real_alpha/train_real_sae.py`와 `/Users/jihoonkwon/Desktop/Projects-old/lvlm_hallucination/src/training/callbacks.py`를 참고했다. 새 SAE는 Hugging Face `PreTrainedModel`로 구현해 저장·불러오기 규약과 `Trainer`를 사용한다. 한 개의 입력이나 상수 입력으로 이루어진 자료 묶음은 분산이 0이므로 나눗셈 분모를 기계 정밀도 이상의 값으로 제한한다. 이는 소규모 점검에서 NaN을 막기 위한 명시적 차이다.

기존 COCO 복원 학습 설정에서 사용하지 않던 죽은 특징 보조 손실은 구현하지 않았다. 보조 손실이나 특징 재활성화를 추가하면 학습 조건이 바뀌므로 별도 설정과 검증이 필요하다.

전체 상관행렬, 활성 특징 선택과 헝가리안 대응은 공개 저장소의 커밋 `63f4289e8bfa2cfa01e4921ae24b06231a88cc20`을 기준으로 확인했다. 상관계수는 부호 있는 피어슨 값이며, 원본에서 한 번이라도 켜진 특징을 연결 대상으로 둔다. 이 구현은 Greedy를 추가하고, 상수 특징의 상관이 정의되지 않는다는 표지를 결과에 함께 남긴다. [원본 상관·대응 코드](https://github.com/JiH00nKw0n/cross_modal_feature_heterogeneity/blob/63f4289e8bfa2cfa01e4921ae24b06231a88cc20/src/alignment/panel.py)

모델은 CLIP ViT-B/32, SAE는 양쪽 각각 4,096개 특징과 TopK 8을 기본값으로 사용한다. AdamW의 학습률 0.0005, 가중치 감쇠 0.00001, 모멘텀 계수 0.9와 0.999, 수치 안정화 상수 0.00000001, 묶음 크기 1,024, 30회 학습, 준비 구간 5%, 코사인 학습률과 최대 기울기 크기 1.0을 설정에 드러냈다. [참조 COCO 설정](https://github.com/JiH00nKw0n/cross_modal_feature_heterogeneity/blob/63f4289e8bfa2cfa01e4921ae24b06231a88cc20/configs/post_rebuttal/clip_b32_coco.yaml)

참조 저장소의 COCO 기본 자료는 Karpathy 분할의 학습 이미지 113,287장이고 캡션마다 이미지 표현을 반복하는 쌍 자료다. 새 기본값은 공식 COCO2017 학습 이미지 118,287장과 모든 캡션이다. 이미지 SAE는 이미지당 한 행을 사용하므로 이미지 쪽의 자료 가중치와 학습 갱신 횟수가 달라진다. `paired_repeat_image` 옵션에서는 캡션별로 이미지를 반복해 두 SAE 손실의 평균을 한 `Trainer`에서 학습한다. 두 모델의 매개변수는 이 옵션에서도 독립적이다. [참조 자료 추출 코드](https://github.com/JiH00nKw0n/cross_modal_feature_heterogeneity/blob/63f4289e8bfa2cfa01e4921ae24b06231a88cc20/src/data/extract.py)

CLIP 가중치는 Hugging Face의 고정 리비전 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`을 사용한다. 라이브러리는 `transformers==4.50.3`과 `torch==2.6.0`을 고정했다. 이미지 전처리는 기존 느린 처리 구현을 명시적으로 사용하며, 이미지·텍스트 투영 벡터는 float32로 길이를 정규화한다. [CLIP 모델 파일](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268), [Hugging Face Trainer 문서](https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/trainer)

연구 질문의 근거는 원래 연구 문서와 이미지·텍스트에 별도 SAE를 학습해 사후 대응을 구하는 선행 연구다. 객체별 대표 선택을 자연적인 존재·부재 분류에서 원본·제거 AUROC로 바꾸고, 가중치를 고정한 입력 편집으로 오류를 검사하는 것이 이번 실험의 변경점이다. [연구 문서](https://project.jihoonkwon.info/f4750a4b/multimodal-SAE/2026-08-31.html), [참조 논문](https://arxiv.org/abs/2606.29888)
