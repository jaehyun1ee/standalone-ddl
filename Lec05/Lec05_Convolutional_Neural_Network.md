# Lec05_Convolutional Neural Network

## Contents

1. Back to Image Classification
2. Convolutional Neural Network

## From Last Lecture,

지난번 강의에서는, Linear Classifier가 Nonlinear Classification Problem에 적용될 수 없다는 한계를 파악했습니다.<br/>

그 한계를 넘어서기 위해, Input Feature를 다른 차원에 Mapping시켜 Linearly Separable하게 만들어준 뒤, 그것에 다시 Linear Classifier를 적용하는 방법을 배웠습니다. 
그리고 이것을 MLP, 혹은 ANN이라 부르기로 했습니다.<br/>

ANN은 인간의 신경망을 모방하여 만든 Model이었습니다. 신경계를 모방하고자, 신경계의 Building Block인 Neuron을 모방한 Perceptron Model을 고안했고, 그러한 Perceptron들을 층으로 쌓은 후 연결시켜 MLP를 구성한 것이었죠!<br/>

DL의 비전처리 분야에서는, CNN이 주로 사용됩니다. 
ANN의 변형된 한 형태이죠. 오늘 수업에서는,
CNN의 원리에 대해 배운 후, 그것을 코드로 옮겨 보는 시간을 가져보겠습니다!

## 1. Back to Image Classification

첫 수업에서, 바로 Image Classification을 다루기는 어려우니, 먼저 Iris Classification을 다루겠다고 했습니다. 
그렇게 Iris Dataset에 대해 K-NN Model도 만들어 보고, Linear Classifier Model도 만들어봤죠?<br/>

이제는 우리의 최종 목표인 Image Classification으로 돌아올 때가 되었습니다. 
지난 시간에 MNIST를 소개하며 Image Classification에 발을 살짝 담궈 봤었죠? 
오늘부터는 Image Classification에 집중해 보겠습니다.<br/>

### A. Pixels

Image는 **Pixel**로 이루어져 있습니다. 
Pixel은 Picture Element로, 이미지를 구성하는 기본 단위입니다.
하나하나의 Pixel은 아주 작은 정사각형인데요, 이러한 작은 정사각형들이 모여서 우리가 보는 Image를 만드는 것이죠!<br/>

먼저, 지난 시간에 봤던 MNIST와 비슷한 Data를 볼까요?<br/>

![Grayscale S](*)

사진은, S라는 손글씨입니다. 
이 흑백 사진에서, 각 Pixel의 밝기는 0 ~ 255의 정수로 표현됩니다. 
숫자가 높을수록 밝은 것이죠. 
따라서, 이 사진의 정보는 왼쪽의 Matrix로 나타낼 수 있습니다.<br/>

이제 흑백이 아닌 사진을 볼까요?<br/>

![RGB](*)
![CIFAR-10](*)

이 경우에는, 각 Pixel이 밝기 뿐만 아니라 색깔도 표현하고 있어야겠죠?<br/>

그래서 RGB라는 3개의 Channel로 색깔을 표현합니다. 
각 Pixel은, (R, G, B)의 Tuple이며, 각각의 값은 0 ~ 255의 정수입니다.
이들 조합에 따라 Pixel이 나타내는 색깔이 달라지는 것이죠. 
그렇다면, 이 사진의 정보는 Tensor로 나타낼 수 있습니다. 
R에 대한 Matrix, G에 대한 Matrix, B에 대한 Matrix가 수직 방향으로 쌓인 것이죠!

### B. Image Classification with MLP

지난 코딩 실습 시간에는 MNIST Dataset을 MLP에 학습시켰습니다.<br/>

학습시키기 위해서 28 * 28 Matrix로 된 손글씨 사진을 784차원의 Vector로 펴 주었습니다. 
그리고, 90%대의 Validation Accuracy를 얻을 수 있었죠!<br/>

CIFAR-10 Dataset도 MLP에 적용해 보면 어떨까요?<br/>

CIFAR-10의 Image는 32 * 32 크기의 컬러 사진입니다. 
따라서, 사진을 32 * 32 * 3 = 3072차원의 Vector로 펴준 다음, MLP에 Feed해주면 됩니다.
Keras 라이브러리로 MLP를 만들어서 학습을 시켜 보면, Accuracy가 그리 높지 않다는 것을 관찰할 수 있습니다. 
제가 직접 해본 결과, 10%의 정확도가 나왔습니다. 
CIFAR-10에서는 Label이 10개이므로, 찍은 것과 비슷한 정확도인 셈이죠. 
Hyperparameter를 적절히 Tuning했다면, 더 좋은 결과가 나왔을 수도 있지만, 아마 50%를 넘기기는 힘들 것입니다.<br/>

MLP 그 자체로는 Image Classification에 적합하지 않은 것 같죠? 
그 이유는 다음과 같습니다.<br/>

**1. Parameter가 너무 많다.** <br/>

CIFAR-10은 3072개의 Input Feature를 다뤘었죠? 
이것을 Perceptron P개짜리 Hidden Layer에 Mapping한 다음, 10개짜리 Output Layer로 보내려면, 총 3072 * P + P * 10개의 Parameter가 필요합니다. 
학습의 대상이 되는 Parameter가 너무 많아지면, 효율적인 학습이 불가능합니다. 
따라서, 학습이 제대로 이뤄지지 않습니다.<br/>

숫자가 너무 작은 것 같다고요? 
하지만, 우리가 실제로 보는 Image는 훨씬 크다는 것을 기억해야 합니다.
32 * 32 크기의 Image는 사실 눈을 아주 가까이 가져가야 보이는 크기입니다. 
제 아이패드의 크기가 2732 * 2048이니까요! 
고해상도의 Real World Problem에 대해 MLP를 적용하려면, Parameter가 엄청나게 많이 필요하겠죠?<br/>

**2. Image를 Flatten하며 Spatial Structure가 손실된다.** <br/>

우리가 사진을 볼 때는, Pixel 하나만을 보는 것이 아니라, Pixel 여러개가 뭉친 하나의 영역을 보고 판단을 내립니다. 
비슷한 색깔의 Pixel들이 모여서 한 덩어리를 만들기 때문이죠. 
하지만, Image를 Vector로 Flatten하는 순간, Pixel들의 공간 정보가 사라집니다. 
Matrix 형태일 때는 이웃했던 Pixel들이 Vector가 되는 순간, 서로 멀리 떨어지게 되니까요.<br/>

이렇게 Image를 Vector로 펴면, 각각의 Pixel을 독립적으로 다루게 됩니다. 
Pixel간의 관계는 따지지 않게 되는 것이죠. 
이것은 우리가 Image를 인식하는 방법과는 다르죠?<br/>

이러한 이유들 때문에, Image Classification에 MLP를 그대로 적용하지는 않습니다. 
MLP를 어느정도 변형해 주어야 합니다.<br/>

우리가 원하는 Model은, MLP보다 적은 Parameter를 쓰면서, Spatial Structure를 보존하는 Model입니다. 
이러한 Model을 어떻게 만들 수 있을까요?<br/>

> :question: 지난 시간에 배운 Universal Approximation Theorem에 의하면, Hidden Layer 1개를 가진 MLP는 어떠한 함수도 근사할 수 있다고 하지 않았나요? 
그러면, Image Classification도 MLP로 가능해야 하는 것 아닌가요?<br/>

> :point_right: _“A feedforward network with a single layer is sufficient to represent any function, but thelayer may be infeasibly large and may fail to learn and generalize correctly.”_ 가 가장 좋은 대답인 것 같네요. 
Universal Approximation Theorem은 매우 강력한 정리이지만, 어디까지나 이론상 가능하다는 것을 증명한 것입니다. 
이론적으로는, Image Classification도 가능하겠지만, 수많은 Parameter들을 실제로 Update시켜주기에는 무리가 있을 것입니다. 
따라서, MLP에 변형을 가해 Image Classification을 실제로 잘 수행하는 Model을 만들고 싶은 것이죠!

## 2. Convolutional Neural Network

### A. Idea

위의 문제들을 해결하기 위해 나온 Model이 바로 **CNN(Convolutional Neural Network, 합성곱신경망)** 입니다. 
CNN이 어떻게 생겼나 그림으로 먼저 알아볼까요?<br/>

![CNN Architecture](*)

이게 무슨 그림인가 싶으시죠? 
지난 시간까지 봤던 MLP랑은 무언가 다릅니다.<br/>

뭐가 다른지를 살펴볼까요?<br/>

우선, Perceptron들이 뒤에 가서야 등장합니다. 
그 전에는, Input Image가 계속 Tensor들로 Mapping이 되고 있습니다.<br/>

또, Layer들에 생소한 이름이 붙었습니다. 
**Convolutional Layer**, **Pooling Layer**, **Fully-Connected Layer**이라는 층들이 등장했습니다.<br/>

Tensor에서 Tensor로 갈 때, 무슨 일이 일어나고 있는 것일까요?<br/>

새롭게 등장한 Layer들의 역할은 무엇일까요?<br/>

또, 이 Model은 Parameter의 수를 얼마나 줄이면서, Image의 공간 정보에서 특징을 뽑아내고 있는 것일까요?<br/>

하나하나 알아보도록 하겠습니다!

### B. Convolutional Layer

Tensor를 Tensor로 Mapping하는 과정에서, 어떻게 공간 정보를 보존할 수 있을까요?<br/>

CNN에서는 Input Tensor에 **Filter**라는 것을 적용하여 공간 정보를 보존합니다. 
우리가 SNOW, Foodie, SODA, Ulike 등의 사진 어플리케이션에서 접하는 Filter가 바로 이 Filter입니다. 
Filter가 어떻게 적용되는가를 볼까요?<br/>

![Filter](*)

위 그림의 파란색 상자가 하나의 Filter인데요, 이 Filter가 Input Tensor의 초록색 영역에 적용되고 있는 모습입니다. 
파란색 Filter와 초록색 영역의 어떠한 연산을 통해 하나의 빨간색 Scalar 값이 나옵니다. 그리고 이 Scalar 값을 Activation Function에 통과시킵니다.<br/>

그리고, 이 Filter가 Tensor 위를 움직이며 Tensor의 영역들에 모두 적용됩니다. 
그러면, Filter를 통과한 Input Tensor는, 하나의 Matrix로 Mapping되겠죠? 
이렇게 Mapping된 Matrix를 **Activation Map**이라고 합니다.<br/>

어떤 연산을 통해 Tensor의 일부, Filter가 Scalar 하나로 Mapping되는지 보여드리기 위해 예시를 가져왔습니다.<br/>

![Filter Demo](*)

위의 그림은 Filter에서의 연산을 나타낸 것입니다. 
Input Tensor의 형태는, 4x4x3이죠? 
이제 이 Tensor에 3x3x3 Filter가 적용됩니다. 
Filter가 Tensor의 왼쪽 위에 적용될 때를 볼까요? 
그러면, Filter가 적용되는 Input Tensor의 한 부분의 형태는 3x3x3이겠죠? 이는 Filter의 크기와 같습니다. 
그러면 두 Tensor를 Align해놓았다고 생각하고, Elementwise하게 곱해준 다음, 그 합을 구하면 하나의 Scalar 값이 나옵니다. 
이 경우에는, 7이 되겠죠. 
이제 이 값을 Activation Function에 통과시키면 최종 Mapping 결과가 나옵니다. 
이 과정을 **합성곱**을 취한다, 혹은 **Convolution**이라고 합니다.<br/>

이런 식으로, Filter가 Input Tensor의 다른 영역들을 덮으면서 결과값들을 계산합니다. 
그 결과 Tensor가 Matrix로 Mapping되었죠?<br/>

그런데, 아까 본 그림에서는 Tensor가 Tensor로 Mapping되었습니다. 
Matrix 여러개가 나와야 한다는 뜻인데, 그러려면 어떻게 해야 할까요?<br/>

Filter 1개가 Tensor를 Matrix로 Mapping하고 있으므로, Filter 여러개를 쓰면 됩니다! 
Filter 여러개를 Input Tensor에 각각 적용해 나온 Activation Map들을 쌓아서 하나의 Output Tensor를 만드는 것이죠!<br/>

이제 Filter들이 Tensor를 Tensor로 Mapping한다는 것은 알았습니다. 
하지만, 이것이 어떻게 공간 정보를 읽으면서 그 특징을 뽑아낸다는 것일까요?<br/>

예시를 통해 살펴보겠습니다.<br/>

편의를 위해, Grayscale Image를 다루겠습니다. 
여기서 0은 흰색, 1은 검은색으로 정의하고, Activation Function은 RELU로 설정하겠습니다.<br/>

![Filter Ex](*)

다음과 같은 Filter가 Image에 적용된다고 할때, 이 Filter의 역할은 무엇일까요? 
우리가 Input을 Mapping할 때, 결국 원하는 것은 Input의 특징들을 뽑아내서 그 특징들을 기반으로 분류하는 것입니다. 
그렇다면, 이 Filter가 뽑아내고자 하는 Image의 특징은 무엇일까요?<br/>

Filter가 1번 사진에 적용되었다고 해봅시다. 
1번 사진은 Major Diagonal이 그려져 있습니다. 
Activation Map은 (3, 0, 0, 3)이 나오죠?<br/>

Minor Diagonal이 그려진 2번 사진에 Filter를 적용하면, Activaiton Map은 (0, 1, 1, 0)입니다.<br/>

단순히 Activation Map의 숫자들의 크기만 가지고 비교하면, Filter에 대해 더 잘 반응하는 사진은 1번 사진이죠?<br/>

Filter가 대각선 방향으로는 1을 곱하고, 나머지에 대해서는 0을 곱하기 때문일 것입니다. 
따라서, 이 Filter는 Image 안의 Major Diagonal을 찾는 역할을 합니다.<br/>

비슷한 방법으로, 직선, 원 등의 특징을 찾아내는 Filter들도 있을 것입니다. 
이들 Filter를 Input Image에 적용해, Image의 특징을 내포한 Tensor로 Mapping시키는 것이죠.<br/>

특징을 잘 뽑아내려면, Filter의 값들이 잘 설정되어야겠죠? 
Filter의 값들이 바로 CNN에서의 Parameter입니다. 
학습을 통해서 특징을 잘 뽑아내는 Filter를 찾아내야 하는 것이죠!<br/>

이때, 학습은 Gradient Descent 방식으로, Backpropagation을 통해 이루어집니다. 
그 Gradient를 구하는 방법은, [[Back Propagation in Convolutional Neural Networks - Intuition and Code](https://becominghuman.ai/back-propagation-in-convolutional-neural-networksintuition-and-code-714ef1c38199)]에 잘 설명되어 있습니다.<br/>

![CNN Architecture](*)

이제 다시 CNN을 봅시다. 
Tensor가 Tensor로 Mapping되는 과정에서 공간 정보에서 얻은 특징들을 뽑아내고 있다는 것을 이해하실 수 있으신가요?<br/>

이제, Filter에 사용되는 몇가지 Terminology를 짚고 넘어가겠습니다.<br/>

> **Stride**

Filter를 Image 위에서 Slide시킨다고 했는데, 이때 Filter가 한번에 몇칸씩 이동해야 할까요? 한칸씩 옆으로 갈 수도 있고, 두 칸씩 옆으로 갈 수도 있습니다.<br/>

Filter가 Slide당 움직이는 칸 수, 혹은 Pixel 수를 **Stride**라고 합니다. 
그러면, Filter의 크기와 Stride의 크기에 따라 Output의 크기가 달라지겠죠? 
Stride가 크면, 그만큼 계산되는 Scalar 값들의 수가 적어질 테니까요.<br/>

Filter의 크기, Stride로 Output의 크기를 계산할 수 있는데요, 그 수식은 아래와 같습니다.<br/>

![Stride](*)

> :question: Image가 정사각형이 아닌 직사각형 모양일 때는 어떻게 계산하나요? 또, Filter가 직사각형인 경우는 어떻게 계산하나요?<br/>

> :point_right: 위 식을 적절히 변형하면 계산할 수 있습니다. 
하지만, 우리는 CIFAR-10에서 정사각행렬 Data를 다루기 때문에 오늘 수업에서 사용될 Notation들은 모두 정사각행렬을 기초로 했다고 이해하시면 됩니다.

> **Padding**

아까 Stride를 설명하며 나온 수식을 다시 볼까요?<br/>

수식을 자세히 보면, Output의 크기가 항상 Input보다 작아짐을 관찰할 수 있습니다. 
(1x1 크기의 Filter는 공간 정보와는 상관없기 때문에 사용하지 않습니다. 따라서, O < I가 되겠죠?) 
그 이유는, 가장 귀퉁이에 있는 Pixel들에 대해서는 FIlter가 Center Align되지 않기 때문이죠. 
Data가 어느정도 손실되고 있는 것입니다!<br/>

따라서, 귀퉁이의 Pixel을 중심으로도 값을 계산해 줄 방법을 찾아야 합니다. 
그러기 위해, 우리는 Image에 **Padding**을 더해 줍니다. 
겨울에 입는 패딩을 생각하시면 될 것 같아요. 
주변에 0이라는 값들을 붙여 줍니다. 
그러면, 이제 귀퉁이 Pixel에 대해서도 Filter를 Center Align할 수 있겠죠?<br/>

다만, Padding이 만병통치약은 아닙니다. 
0이라는 Padding을 넣어줌으로써 원본 Data에 Noise가 더해졌기 때문이죠. 
따라서, Padding을 더할지 말지 결정하는 것도 하나의 Hyperparameter가 되겠습니다.<br/>

이제, 아까 수식에 Padding을 반영해 주면, 아래와 같은 수식이 나옵니다.<br/>

![Padding](*)

## C. Pooling Layer

Filter를 통해 Tensor를 Tensor로 Mapping할 수 있게 되었습니다. 
이 Tensor에는 꼭 필요한 정보도 있겠지만, 분류에 그다지 필요없는 정보도 있을 수 있습니다. 
사람 얼굴을 분류한다고 생각해 봅시다. 피부를 표현한 Pixel들이 넓게 퍼져 있을텐데, 이것들이 모두 필요한 정보들일까요? 
사실 피부색만 알고 있으면 충분합니다. 
따라서, Tensor에서 필요없는 정보들을 쳐낼 필요가 있습니다.<br/>

Tensor에서 필요없는 정보들을 쳐내는 과정을 **Pooling**이라고 합니다. 
Tensor의 각 영역에서 대푯값 하나를 뽑아내어, 그 크기를 줄여 주는 것이죠.<br/>

대푯값을 정의하는 방법에는 여러 가지가 있습니다. 
최댓값을 뽑아내는 **Max Pooling**, 평균값을 뽑아내는 **Average Pooling** 등등이 사용됩니다.<br/>

그림으로 보면 아래와 같습니다.<br/>

![Pooling Demo](*)

4x4 Activation Map을 2x2 영역들로 나눠서 Pooling을 했더니, 2x2 Matrix가 됐습니다. 
원본 Data에서 대푯값을 뽑아내, Data의 크기를 줄인 것이죠.<br/>

또, 여기서 눈여겨 볼 것은, 아까 Filter와는 다르게 Parameter가 개입되지 않았다는 것입니다! 
아까 Filter의 값들은 Parameter로, 학습의 대상이었으나, 여기서는 그저 최댓값, 평균값을 기준으로 값을 뽑아내고 있으므로, Parameter가 없는 것이죠.<br/>

Pooling도 마찬가지로, Input에 대해 Output의 크기를 계산해 줄 수 있습니다.  
그 수식은 아래와 같습니다.<br/>

![Pooling](*)

일반적으로, Pooling의 Stride는 Pool Size와 동일합니다. 그것을 가정하고 만든 수식입니다!

### D. Fully-Connected Layer

지금까지 Tensor를 Filter를 통과시켜 Tensor의 공간정보를 포함한 특징을 뽑아내는 것을 배웠고, Pooling을 통해 그 크기를 줄여 주는 방법을 배웠습니다.<br/>

공간 정보를 포함한 특징을 뽑아냈고, 그 특징의 개수도 많이 줄었으니, 이제 분류에 MLP를 적용하면 됩니다. 
Filter, Pooling을 적용해 최종적으로 나온 Tensor를 Flatten하여 MLP의 Input으로 주는 것이죠. 
이렇게 Tensor가 Vector로 변하는 층을 **Fully-Connected Layer**라고 합니다.<br/>

FC Layer를 통해서 Flatten된 Data에 MLP를 적용해서 최종 Classification을 하는 것이죠!<br/>

### E. CNN Architecture

아까 본 CNN Model을 하나하나 뜯어봤습니다.<br/>

**Convolution Layer**를 통해서 Tensor의 공간정보를 포함한 특징을 뽑아내고,<br/>

**Pooling Layer**를 통해 Tensor의 특징을 유지하며 크기를 줄여줍니다.<br/>

그리고 **Fully-Connected Layer**를 통해 뽑아낸 특징들을 MLP에 Feed하여 Classification을 하는 것이죠.<br/>

이제 아까 제시했던 질문이 딱 한개 남았습니다. 
CNN이 MLP에 비해 Parameter의 개수가 더 적을까요?<br/>

우리의 CNN Model을 따라가며 한번 계산해 보겠습니다.<br/>

| Layer | Input Size | Output Size | # Parameters |
|:-:|:-:|:-:|:-:|
| CONV1 | 32x32x3 | 32x32x32 | (5 * 5 * 3) * 32 = 2400 |
| POOL1 | 32x32x32 | 16x16x32 | 0 |
| CONV2 | 16x16x32 | 16x16x16 | (5 * 5 * 32) * 16 = 12800 |
| POOL2 | 16x16x16 | 8x8x16 | 0 |
| CONV3 | 8x8x16 | 8x8x32 | (5 * 5 * 16) * 32 = 12800 |
| POOL3 | 8x8x32 | 4x4x32 | 0 |
| FC | 4x4x32 | 512 | 0 |
| Softmax | 512 | 10 | 512 * 10 = 5120 |

총 33120개의 Parameter가 사용됩니다.<br/>

만약 이 Model을 MLP로 만들었다면 어떨까요?<br/>

CONV + POOL의 한 세트를 거친 결과를 한 Hidden Layer라고 생각하면, MLP의 구조는,<br/>

| Layer | Input Size | Output Size | # Parameters |
|:-:|:-:|:-:|:-:|
| Input | - | - | - |
| Hidden1 | 32x32x3 = 3720 | 16x16x32 = 8192 | 3720 * 8192 = 30474240 |
| Hidden2 | 8192 | 8x8x16 = 1024 | 8192 * 1024 = 8388608 |
| Hidden3 | 1024 | 4x4x32 = 512 | 1024 * 512 = 524288 |
| Output | 512 | 10 | 512 * 10 = 5120 |

총 35107840개의 Parameter가 사용됩니다.<br/>

CNN이 MLP보다 훨씬 적은 수의 Parameter를 사용한다는 것을 알 수 있죠?<br/>

### F. Performance

아까 MLP에 Image Classification을 바로 적용했을 때의 문제를 해결하기 위해 CNN을 만들었다고 했습니다. 
그러면, CNN이 실제 Image Classification에 대해 좋은 분류 결과를 줄까요?<br/>

그렇습니다!<br/>

CNN을 아주 많이 응용해서 만든 Model들은 95% 정도의 정확도를 기록하고 있고, 지금 배운 CNN을 적절히 적용하면 80%정도의 정확도를 얻을 수 있습니다.
CIFAR-10에 대한 정확도 랭킹은 [[Classification Datasets Results](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)]에서 확인할 수 있습니다!<br/>

오늘 코딩실습 시간에는 Keras로 CIFAR-10 Dataset을 분류하게 될텐데요, 지금은 DL의 다양한 기법을 배우지 않은 상태이기 때문에 아마 50% 정도의 정확도가 나올 것입니다. 
하지만, 아까 MLP에서 나온 10%보다는 훨씬 높은 값이죠?<br/>

또, DL 기법들을 배우고 나면 Colab으로도 정확도 80%짜리 Model을 만드실 수 있을 겁니다. 
그러니, 오늘 결과를 가지고 너무 실망하지 마세요!

## Preview on Next Lecture(s)

다음 수업부터는, 지난 시간에 살펴보았던 MLP의 문제점을 개선하는 방법을 다룰 예정입니다. 
Overfitting을 어떻게 방지하는지, 어떤 Update Algorithm이 우수한지 등등을 다룰 것입니다. 
배운 내용들을 적용할수록 더 정확한 Model을 만들 수 있겠죠?<br/>

그리고, 아까 CNN Model을 응용해 만든 Model들이 있다고 했습니다. 
이러한 Model들은 어떤 형태인지를 또 다룰 예정입니다!
