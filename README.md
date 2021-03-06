# Alzheimer's disease diagnosis by CNN classifier
* 뇌 MRI를 이용한 알츠하이머 진단 모델
   - ADNI 제공 데이터셋 이용
   - Data Augmentation 적용(Crop)
   - 3D CNN 이용   
   
   
### MRI 기반 알츠하이머병 진단을 위한 CNN 모델의 복잡도와 성능의 연관성 분석
Empirical analysis of the relation between the complexity and accuracy of CNN models for the diagnosis of Alzheimer’s disease based on MRI - 전민경, 이현석, 김성찬

- MRI 기반 알츠하이머병 진단 사례에서 적당한 복잡도의 모델에 데이터 augmentation을 적용해 과적합을 줄이는 것이 우수한 성능으로 이어진다는 것을 실험적으로 분석한다.
- 분석을 위해 다양한 복잡도를 가진 CNN 모델들에 데이터 augmentation 기법을 적용해 과적합을 최소화 했을 때 모델들의 성능을 검증한다.
- 분석 결과 데이터 augmentation 기법에 의해 과적합이 해결될 수 있는 기본 CNN 모델이 높은 복잡도의 ResNet 모델보다 우수한 진단 성능을 보였다.
- 높은 복잡도가 높은 성능으로 이어진다는 일반적인 인식과 달리 의료 영상 진단에서는 학습데이터의 특징에 따른 모델 전정이 필요함을 의미한다.
   
   
### 데이터
* Alzheimer’s Disease Neuroimaging Initiative(ADNI)-1의 standardized MRI 데이터셋 중 screening 데이터 사용
    - 818개 : 정상 229개, 경도 인지 장애 401개, 알츠하이머 환자 188개
* 데이터 전처리
   - 공간적 정규화 & 배경 제거 & 복셀 값 정규화[0,1]
* Data augmentation
   - MRI는 개별 데이터가 크고 전체 수는 적기 때문에 모델 과적합 해소를 위해 Crop 기법 사용
   - train : 각 꼭짓점과 중앙, 총 9개 중 random crop
   - test : 각 꼭짓점과 중앙에서 crop
   
### 모델 구조
* ResNet, VGGNet, Inception, Networks In Networks(NIN)에 기반
   - ResNet : Stem layers
   - VGGNet : 특징 맵 수가 증가할 때마다 각 특징 맵 사이즈를 줄임
   - Inception : 1x1x1 convolution
   - Network In Network : global average pooling
   
   
### 실험 결과

   
   
### 결론
- MRI를 이용한 알츠하이머병 진단 사례에서 모델의 최적 복잡도가 매우 제한적인 학습 데이터의 과적합 해소에 의해 결정됨
- 일반적인 자연 영상 기반 문제와 달리 의료 영상에서 간단한 모델이 복잡한 모델보다 좋은 성능을 보이는 것을 실험적으로 검증
- 의료 영상을 분석할 때는 주어진 학습 데이터의 특징에 따라 적합한 데이터 augmentation 기법의 적용과 이에 의해 과적합이 해결될 수 있는 정도의 복잡도를 가진 모델이 조합될 때 최적의 진단 성능을
보일 것이다
