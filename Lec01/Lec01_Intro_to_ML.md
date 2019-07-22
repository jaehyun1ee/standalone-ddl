# Lec01_Intro to ML

## Contents

1. What is Machine Learning?
2. Making a Model I
3. Testing a Model
4. Making a Model II

## 1. What is Machine Learning?

### AI vs. ML vs. DL

AI(Artificial Intelligence, 인공지능), ML(Machine Learning, 기계학습), DL(Deep Learning, 딥러닝)은 서로 다른 분야일까요?
비슷한 의미로 사용되는 단어들 같은데, 서로 어떤 관련이 있을까요?<br/>

AI의 범주 안에 ML이 포함되고, ML의 범주 안에 DL이 포함됩니다.<br/>

> AI

먼저, AI는 인간의 지능을 모방한, 지능형 기계를 뜻합니다. 
따라서, 정수 x에 대해 양수와 음수를 구분하는 아래의 코드도 하나의 인공지능이라 할 수 있습니다.<br/>

```
  if x >= 0:
    print(“POSITIVE”)
  else:
    print(“NEGATIVE”)
```

위의 코드는 정수의 음/양을 판단하는 지능을 explicit하게 코드로 옮긴 결과입니다.<br/>

> ML

ML은 다음 문장으로 설명 가능합니다.<br/>

```
  “The field of machine learning is concerned with the question of how to construct computer programs that automatically improve with experience.”
```

즉, 경험을 통해 학습하고, 발전하는 컴퓨터 프로그램을 만드는 것이 ML의 목표입니다. 
ML의 학습 알고리즘을 통해, 기계는 지능을 implicit하게 내장하게 됩니다. 
위에서 다룬, 정수의 음양을 구분하는 프로그램처럼 지능이 겉으로 드러나지 않는 거죠!

> DL

DL은 말그대로, 더 깊은 ML을 의미합니다. 
DL에서는 복잡한 데이터 처리를 위해, 인간의 신경망을 모방한 ANN(Artificial Neural Network, 인공신경망)을 사용합니다. 
4회차에 DL과 ANN이 등장할 테니, 그때 더 자세한 설명을 붙이겠습니다.

### Fields of ML

ML의 분과는 크게 Supervised Learning(지도학습)과 Unsupervised Learning(비지도학습)으로 구분 가능합니다. 
ML에서는 기계에게 지능을 학습시킨다고 했는데요, 기계를 학습시키기 위해서는 데이터가 필요합니다. 
이때, 어떤 데이터로 기계를 학습시키냐에 따라 Supervised Learning, Unspervised Learning으로 나뉩니다. 
Supervised Learning의 경우, 정답을 기계에게 알려주며 학습시키는 반면, Unsupervised Learning에서는 기계에게 정답을 알려주지 않고 학습시킵니다. 
뒤의 예시를 보면 둘의 차이가 더 분명히 이해될 것입니다.<br/>

또한, 학습의 결과가 Discrete한지, Continuous한지에 따라서도 구분이 가능합니다.<br/>

각각에 대해 더 설명하면,<br/>

> Supervised

>> Discrete : classification or categorization

기계는 분류 방법을 학습합니다. 
예를 들어, 기계에게 고양이 사진을 주고, “이것은 고양이 사진이야.” 라고 정답을 알려줍니다. 
또, 기계에게 사자 사진을 주고, “이것은 사자 사진이야.” 라고 정답을 알려줍니다. 
이런 식으로 기계를 학습시킨 후, 기계는 입력으로 들어온 사진에 대해, 그 사진이 어떤 동물인지 분류할 수 있게 됩니다.

>> Continuous : regression

기계는 회귀 식을 학습합니다. 
예를 들어, 기계에게 학생들의 공부 시간과 그에 따른 기말고사 점수를 학습시킵니다. 
학습 이후, 기계는 공부 시간과 기말고사 점수 사이의 관계식을 학습하게 되어, 10시간 공부한 학생의 기말고사 점수는 몇점일지 예측할 수 있게 됩니다.

> Unsupervised

>> Discrete : clustering

학습 데이터로 정답이 없는 데이터가 주어집니다. 
기계는 서로 비슷한 데이터를 군집화하는 방법을 학습합니다. 
예를 들어, 정답 없이 고양이 사진, 사자 사진, 강아지 사진, 말 사진, 돼지 사진들을 주면, 기계는 서로 비슷한 사진들을 하나의 class로 묶습니다. 
학습이 잘 되었다면, 고양이 사진은 class 1, 사자 사진은 class 2 와 같이 군집화가 될 것입니다.

>> Continuous : dimensionality reduction

이것 또한 학습 데이터로 정답이 없는 데이터가 주어집니다. 
기계는 주어진 데이터에서 feature(특징)들을 뽑아서 데이터의 특징을 보존하면서, 데이터의 차원을 줄여 줍니다. 
예로는, PCA(Principal Component Analysis, 주성분분석)과 AE(Autoencoder)가 있습니다.

### Narrow Down to Image Classification

위에 설명한 ML의 분과들을 모두 다루기에는 우리에게 주어진 시간이 부족합니다. (사실, 전달할 역량도 없습니다.)<br/>

따라서, 이번 특강은 Classification에서도 특히 Image Classification을 다룰 것입니다. 
특강의 목표는, 사진 속의 물체가 어떤 class에 속하는지를 예측하는 프로그램을 만드는 것입니다.<br/>

대략 [[Live Classifier](http://cs231n.stanford.edu/)]와 같은 분류기를 만드는 것이 목표입니다!<br/>

Image Classifiecation에 대해 먼저 큰 그림을 그리고 시작하겠습니다. Image Classification은 크게 두 단계로 나눌 수 있는데요,

1. Train

정답이 달린 사진들을 프로그램에게 학습시킵니다. 
강아지 사진을 보여주고 “이건 강아지야.”라고 알려주고, 다음 고양이 사진을 보여주고 “이건 고양이야”라고 알려주는 식으로 학습이 진행됩니다. 
이 과정에서, 프로그램은 ‘대강 이렇게 생긴 사진을 강아지/고양이라고 하는구나.’를 학습합니다.

2. Predict

학습된 프로그램에게 정답이 아직 달리지 않은 사진을 줍니다. 
그러면 프로그램은 학습된 정보를 바탕으로, 그 사진이 어떤 동물의 사진인지를 예측합니다.
이때, 학습의 대상이 되며, 학습 이후 예측을 하는 프로그램을 앞으로 Model이라고 부르겠습니다.<br/>

이제, Model을 어떻게 만드는지 알아보겠습니다.

## 2. Making a Model I

### Narrow Down to Iris Classification

특강의 범위를 Image Classification으로 좁혔지만, 벌써부터 사진을 다루면 개념 이해가 어려울 것 같습니다.
개념의 이해가 가장 중요하므로, 3회차까지는 Image Classification을 다루지 않고, Iris Classification을 다룰 것입니다.<br/>

Iris는 붓꽃인데요, 붓꽃에는 Setosa, Versicolor, Virginica의 세 종이 있습니다. 
이 세 종의 붓꽃들에 대해, 꽃잎의 길이와 너비, 그리고 꽃받침의 길이와 너비를 조사해 만든 Iris Dataset이 준비되어 있습니다.<br/>

Dataset의 A열이 관측 번호이고, B ~ E열이 꽃잎과 꽃받침의 길이와 너비 정보입니다. 
이들을 꽃의 특성, 즉 feature라 부르겠습니다. 
F열은 해당 꽃의 종인데요, 각 특징에 대한 정답이므로, label이라 부르겠습니다.<br/>

우리의 목표는, 주어진 학습 Dataset(Feature와 Label들)을 통해 Model을 학습시켜, 새로운 input feature에 대해 label을 예측하는 것입니다.

### Data-Driven Approach

Model을 만드는 첫번째 방법은, 현재 가지고 있는 Dataset을 그대로 활용하는 방법입니다.<br/>

Train 단계에서는, Model이 Dataset을 전부 외우게 합니다. 
Iris Classification의 경우, 150개의 붓꽃 정보들을 Model의 메모리에 올리는 것이 되겠습니다.<br/>

Predict 단계에서는, 주어진 input feature과 가장 비슷한 Dataset의 data를 찾습니다. 
그리고, 가장 비슷한 data의 label이 input feature의 label일 것이라고 예측합니다.<br/>

여기서, ‘비슷하다’는 추상적인 표현인데요, 프로그램이 예측을 내리기 위해서는 비슷한 정도를 수치적으로 표현해 주어야 합니다. 
이때, feature가 서로 비슷하다는 것을 어떻게 수치적으로 정의할 수 있을까요?

### Nearest Neighbor Algorithm

Nearest Neighbor Algorithm에서는, 두 feature vector 사이의 거리로 두 feature의 비슷한 정도를 나타냅니다.<br/>

N차원 공간상에서의 거리를 구하듯, vector component들의 차이의 제곱합의 제곱근을 구한 것을 L2 distance라 합니다.<br/>

L1 distance는, vector component들의 차이의 절댓값의 합을 구한 것입니다.<br/>

이렇게 구한 L1 distance나 L2 distance가 작을수록, 두 feature가 비슷하다는 뜻이 되겠죠?<br/>

이제, input feature과의 distance가 가장 작은 Dataset의 data를 찾아, 해당 data의 label로 prediction을 내립니다.

### K-Nearest Neighbor Algorithm

K-Nearest Neighbor Algorithm은 위의 Nearest Neighbor Algorithm의 변형입니다.<br/>

NN은 가장 distance가 작은 data 1개를 고르는 반면, K-NN은 distance가 작은 data k개를 고릅니다.
그리고, 그 k개의 data 중에서 과반수의 label로 prediction을 내립니다.<br>

[[K-NN Demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/)] 에서 K-NN Algorithm을 시각화한 결과물을 볼 수 있습니다.<br/>

이렇게 우리의 첫번째 Model이 완성되었습니다. NN과 K-NN을 이용한 Iris Classification은 Github Repo에 올려 두었습니다.
