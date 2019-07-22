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

ML의 분과는 크게 Supervised Learning(지도학습)과 Unsupervised Learning(비지도학습)으로 구분 가능합니다.<br/>

ML에서는 기계에게 지능을 학습시킨다고 했는데요, 기계를 학습시키기 위해서는 데이터가 필요합니다. 
이때, 어떤 데이터로 기계를 학습시키냐에 따라 Supervised Learning, Unspervised Learning으로 나뉩니다.<br/>

Supervised Learning의 경우, 정답을 기계에게 알려주며 학습시키는 반면, Unsupervised Learning에서는 기계에게 정답을 알려주지 않고 학습시킵니다.<br/>

또한, 학습의 결과가 Discrete한지, Continuous한지에 따라서도 구분이 가능합니다.<br/>

뒤의 예시를 보면 분과들의 차이가 더 분명히 이해될 것입니다.

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

## 3. Testing a Model

### Test Set

지금까지 열심히 Model을 만들었습니다. 그러면 이렇게 만든 Model을 바로 Iris Classification에 활용할 수 있을까요?<br/>

아닙니다. 
그 이유는, “뭘 믿고?”라는 질문으로 요약될 수 있습니다.<br/>

우리는 Model을 일단 만들기는 했지만, 이 Model이 얼마나 정확한 Prediction을 내리는지는 모릅니다. 
그렇기 때문에, Iris Classification에서 이 Model의 Prediction을 신뢰할 수 없습니다. 
이 Model의 정확도가 100%라면 좋겠지만, 30%일 수도 있기 때문입니다.<br/>

Iris Classification에 Real-world problem을 대입하여 생각해 보면 이 설명이 더 와닿을 것 같습니다. 
메일링 서비스의 스팸 메일 분류기를 개발하는 부서에 있다고 상상해 봅시다. 
몇달간의 개발로, 스팸 메일 분류기를 완성했습니다. 
회사는 분류기를 100% 신뢰하기로 하고, 분류기를 바로 서비스에 적용했습니다. 
그런데 이때, 알고 보니 분류기가 스팸 메일이 아닌 메일도 스팸으로 잘못 분류하는, 정확도가 낮은 분류기였던 겁니다! 
하지만, 회사는 분류기를 100% 신뢰하기 때문에, 클라이언트들의 메일이 잘못 분류되고 있다는 사실을 인지하지 못합니다. 
끔찍하죠?<br/>

그렇기 때문에, 우리는 우리가 만든 Model이 실제로 Prediction에 사용될 때, 어느 정도의 정확도를 갖는지 Test할 필요가 있습니다.<br/>

따라서, 우리는 주어진 Dataset을 Training set과, Test set으로 구분합니다.<br/>

1. Training set으로는 Model을 학습시키고, 
2. Test set으로는 완성된 Model의 정확도를 측정합니다.

다시 말하면, Test set은 Model이 unseen data에 대해 얼마나 잘 작동하는가를 시험하기 위해 존재하는 거죠!

### Validation Set

K-NN Algorithm으로 Model을 만들 때, 우리는 Model을 오로지 하나만 만들 수 있을까요?<br/>

아닙니다. 
Feature간의 distance를 구할 때, L1 distance를 쓸 수도 있고, L2 distance를 쓸 수도 있습니다. 
또한, K를 3으로 설정할 수도 있고, 5로 설정할 수도 있습니다.<br/>

이러한 선택들은 기계가 학습하기 이전에 사람이 결정해줘야 하는 것들입니다. 
이러한 선택들을 hyperparameter라 부르겠습니다.<br/>

그러면, hyperparameter들의 조합으로 여러 가지 Model을 만들 수 있겠죠?<br/>

Hyperparameter의 조합에 따라, Model의 성능이 좋을 수도 있고, 나쁠 수도 있습니다. 
또한, 일반적으로 잘 작동하는 Hyperparameter 조합은 거의 없으며, 많은 경우 problem-dependent합니다.
따라서, 우리는 다양한 Hyperparamerter의 조합으로, 최적의 Model을 찾아야 합니다.<br/>

그러면, Training set으로 다양한 Model들을 각자 학습시킨 후, Test set에서 측정한 정확도가 가장 높은 Model을 선택하면 되지 않을까요?<br/>

얼핏 보면 맞는 이야기 같지만, 사실 그렇게 해서는 안됩니다. 
Test set은 Model이 unseen data에 대해 얼마나 잘 작동하는가를 시험하기 위해 존재하기 때문이죠.<br/>

우리가 Test set에서의 성능이 가장 높은 Model을 선택한다는 것은, Test set에 대해 가장 잘 작동하는 hyperparameter 조합을 선택한다는 것과 같습니다.
그러면, Test set이 학습 과정에 개입하게 되고, Test set은 unseen data에 대한 performance를 표현할 수 없게 됩니다.<br/>

이 문제를 해결하기 위해, 우리는 주어진 Dataset을 Training set, Validation set, Test set의 총 3개의 부분으로 나눕니다.<br/>

1. Training set으로 다양한 Model들을 각각 학습시킨 후, 
2. Validation set에서의 정확도가 가장 높은 Model을 선택합니다. 
3. 그리고, Test set으로 선택된 Model의 unseen data에 대한 정확도를 계산하는 거죠!

### Cross-Validation

Model을 학습시키고, 최적의 hyperparameter 조합을 찾고, 그 성능을 test하려면, 전체 Dataset을 세 부분으로 나눠야 합니다.<br/>

하지만, 현재 가지고 있는 Dataset이 너무 작다면 어떻게 해야 할까요? 
전체 Dataset을 나누다 보니 Training set과 Validation set이 고르게 나눠지지 않을 수도 있지 않을까요?<br/>

이러한 경우, (K-Fold) Cross-Validation 기법을 사용하여, 작은 Dataset으로도 학습을 충분히 할 수 있도록 합니다.<br/>

1. 전체 Dataset에서 먼저 Test set을 골라 제외해 줍니다. 
2. 그 다음, 남은 Dataset을 총 K개의 fold(조각)으로 나누어 줍니다. 
3. 이제, K개의 fold중 1개가 Validation set, 나머지 (K - 1)개가 Training set을 수행하며, 학습이 이루어집니다.<br/>
모든 fold가 Validation set의 역할을 수행할 수 있도록 총 K번 iterate시키며 학습이 진행되고, K번의 Validation에서의 평균으로 각 Model의 성능을 판단합니다. 
4. 그 다음, 원래 하던대로 Test set을 이용하여 unseen data에 대한 정확도를 측정하면 되겠습니다.

## Making a Model II

### Limitations of K-NN

이제 우리는, Data-Driven Approach로 Model을 만들고, 그것을 Test할 수 있게 되었습니다.<br/>

하지만, 이 방법으로 좋은 Model이 얻어진다면, 9회차짜리 특강을 할 필요가 없었겠죠?<br/>

사실 K-NN Algorithm은 Image Classification에 적용하기에 부적절할 뿐더러, 실제 사용하기도 비효율적이라는 문제가 있습니다.

우선, K-NN은 Image Classification Problem에 부적합합니다.<br/>
K-NN을 Image Classification에 적용하면, 두 feature간의 거리를 구할 때, pixel들의 차이를 계산하게 됩니다.<br/>
이때, 강아지 사진 하나를 학습시키고, 새로운 input으로 좌우 반전된 사진을 주었다고 생각해 봅시다.<br/>
우리는 Model이 강아지라고 예측하기를 바라지만, pixel 단위로 쪼개서 본 두 사진의 distance가 크기 때문에, Model은 두 사
진이 서로 비슷하지 않다고 판단할 것입니다. 이것은 좋은 Model이 아니겠죠?<br/>

또한, K-NN은, Train시킬 때는 전체 Dataset을 메모리에 올리면 되므로, Train이 금방 끝납니다. 
하지만, Prediction을 내릴 때는, 전체 Dataset과 하나하나 비교하여 가장 비슷한 data를 찾아야 하므로, Prediction에 아주 많은 시간이 걸립니다.<br/>

Iris Classification에서는, data가 150개 정도 뿐이고, feature도 4개 뿐이므로, Prediction이 금방 내려지는 것 같지만, real-world problem에서는 훨씬 방대한 Dataset이 쓰입니다.<br/>

우리가 Image Classification에서 다룰 CIFAR-10 Dataset에는 3072개 feature를 가진 사진이 6만개가 있습니다. 
이것들을 하나하나 비교하며 Test하기에는 무리가 있겠죠?<br/>

우리가 실제로 ML을 적용할 때는, Training에는 많은 시간이 걸려도 크게 상관이 없습니다. 
하지만, Prediction을 내릴 때는 그 결과가 금방 나오기를 기대합니다.<br/>

Facebook에서 얼굴 인식 알고리즘을 K-NN으로 구현했다고 상상해 봅시다. 
사진 1개에 대해 얼굴 인식을 하려면 아마 반나절 이상이 걸릴 겁니다. 
결과가 바로 나와야 하는데, 이러면 아주 곤란한 상황이 생길 것입니다.<br/>

따라서, 우리는 이제 Data-Driven Approach를 버리고, 새로운 Approach를 취해야 합니다. 
각 label의 특징을 잘 학습하면서, Prediction을 금방금방 내릴 수 있는 Model을 어떻게 만들 수 있을까요?

### Parametric Approach

전체 Dataset을 외우는 Data-Driven Approach에서 발생하는 문제들을 피하기 위해, 각 label별 특징들을 변수에 저장하고 있으면 어떻까요?<br/>

Train 과정에서는, 변수가 label별 특징들을 잘 내포할 수 있도록 그 값을 조정해 주고, Prediction을 내릴 때는, input과 변수만 가지고 예측을 하는거죠!그러면, Prediction이 금방 내려질 것입니다! 
변수가 label별 특징을 잘 학습하게만 하면 되는거죠.<br/>

우리는 이 변수를 parameter라고 부르겠습니다.<br/>

그러면, 이 접근 방법은, Parametric Approach라고 부르면 되겠네요.

### Linear Classifier

> Idea

그런데, parameter가 label의 특징을 어떻게 가지고 있을 수 있을까요?<br/>

Iris Classification에서, Setosa 종의 꽃잎 길이가 다른 종에 비해 유난히 길다고 가정해 봅시다.<br/>

그러면, 우리는 꽃잎의 길이에 집중해서 꽃들을 분류할 수 있을텐데요, Setosa를 분류할 때, 꽃잎 길이를 더 유심히 관찰하여 분류할 수 있을 겁니다. 
다른 말로 표현하면, 꽃잎의 길이에 가중치를 두는 거죠.<br/>

Linear Classifier는 바로 이 가중치를 통해 데이터를 분류합니다.<br/>

Linear Classifier는 feature를 각 label에 대한 점수로 mapping합니다. 
그 다음, 점수가 가장 높은 label로 Prediction을 내리는데요, mapping 과정에서 parameter가 개입됩니다. 
아까 이야기한 가중치가 숫자로 표현되어 parameter가 되는데요, 아까 Setosa 예시를 이어서 보면, Setosa에 대한 점수를 계산할 때, 꽃잎 길이에 큰 수를 곱해서 점수를 계산하는 식으로 가중치가 반영됩니다.

> Algebraic Interpretation

정리하면, Linear Classifier는, 각각의 feature에 가중치를 곱해서 더한 값으로 각 label의 점수를 계산합니다.<br/>

이러한 계산들을 깔끔하게 식으로 정리하면, s = W * x가 되겠죠.<br/>

W의 각 row는, 각각의 label에 대한 가중치 parameter들이 됩니다.
이 parameter W를 Weight라고 부르겠습니다. 
사실, Linear Classifier는 s = W * x + b로 점수를 계산하는데요, 여기서 더해진 b는 Bias라고 합니다.
Bias가 갑자기 왜 더해졌는지는 뒤에서 설명하겠습니다.<br/>

> Geometric Interpretation

위에서 설명한 식 s = W * x + b는 기하적으로 어떤 의미를 내포하고 있을까요? 왜 이것을 Linear한 분류기라고 부르는 걸까요?<br/>

먼저, 각 data를 기하적으로 보면, Iris Classification problem에서 각 data는 4차원 공간상의 한 점입니다. 
그리고, 이 점들을 label에 따라 색칠을 해서 보면, 같은 label들은 비슷한 위치에 분포되어 있음을 볼 수 있습니다.<br/>

이들을 분류하는 것이 Linear Classifier의 목표인데요, Linear Classifier는 이들을 decision hyperplane으로 구분합니다.<br/>

W * x + b의 각 row를 기하적으로 보면, W의 row가 법선벡터이고, 원점에서 b의 component정도만큼 떨어져 있는 hyperplane입니다. 
W * x + b = 0으로 두고 만든 hyperplane을 보면, 각 label을 분류해 주는 decision hyperplane이 된다는 것을 관찰할 수 있습니다.<br/>

그러면, bias가 왜 필요한지도 유추하실 수 있을텐데요, bias가 없었다면, decision hyperplane들이 모두 원점을 지나야 할 것입니다. 
그러면, 공간상의 label들을 잘 구분하지 못할 것입니다. 따라서, hyperplane들을 띄워 주기 위해 bias가 추가되었다고 볼 수 있습니다.

*** 그림은 4차원 공간을 2차원으로 줄여서 표현했기 때문에, decision hyperplane의 분류 결과가 명확하지 않을 수 있습니다. 
하지만, 4차원 공간 안에서는 decision hyperplane이 각 label들을 잘 분류해 주고 있습니다.<br/>

*** 기하적으로 W * x + b를 그려 보면, decision hyperplane이 되는구나! 정도만 이해하시면 충분합니다.<br/>

이로써, Linear Classifier가 무엇을 하는 Model인지 알아보았습니다.<br/>

하지만, 정확한 분류 결과를 주는 parameter W와 b는 우리에게 주어지는 것이 아니라, 프로그램이 학습을 통해 찾아야 하는 대상입니다. 
따라서, 현재의 부정확한 W와 b를 학습을 통해 정확한 W와 b로 update시켜줄 필요가 있습니다.<br/>

[[Live Linear Classifier](http://vision.stanford.edu/teaching/cs231n-demos/linearclassify/)]를 보시면, Linear Classifier가 update되는 것을 관찰하실 수 있습니다.<br/>

그렇다면, 이러한 학습은 어떻게 이뤄지는 걸까요?<br/>

이러한 학습을 Model Optimization이라고 부르는데요, Optimization을 위해, 두 가지 문제를 해결해야 합니다.<br/>

1. Train이 잘 되었는지 확인할 수치적 척도가 필요합니다.

현재의 Model이 아주 부정확하다면, parameter를 많이 update시켜야 할 것이고, 거의 정확하다면, parameter를 조금만 update시켜줘도 될 것입니다. 
현재 Model이 update되어야 하는 정도를 수치화하여 표현할 필요가 있습니다.

2. Parameter를 update하는 algorithm이 필요합니다.

현재의 parameter가 부정확하다면, 어떻게 해야 그 값을 정확하게 바꿀 수 있을까요?<br/>

우리는 2회차와 3회차에 걸쳐 위 질문들을 답하며, Model Optimization을 공부하겠습니다.<br/>

2회차에서는 Train이 잘 되었는지 확인할 수치적 척도를 만들 것인데요, 현재 parameter의 부정확한정도를 Loss라고 정의할 것입니다. 
이 Loss를 어떻게 계산하는지를 다음 시간에 다루겠습니다.
