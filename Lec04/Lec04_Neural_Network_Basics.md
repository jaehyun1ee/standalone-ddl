## Contents

1. Limitations of Linear Classifier
2. Perceptron
3. MLP
4. Limitations of MLP

<hr/>

## From Last Lecture,

지난 시간까지는, Linear Classifier를 통한 Iris Classification을 해봤습니다.<br/>

우선, Feature들을 Label별 점수로 Mapping하는 Linear Classifier를 만들었죠. 
이때, Mapping 과정에 Parameter W, b가 사용되었습니다.<br/>

그 다음, Backpropagation을 통한 Gradient Descent로 Parameter W, b를 적절하게 Update시켜 주었습니다.<br/>

하지만, 지난 강의를 마치며, Linear Classifier Model이 Image Classification에 있어서 좋은 Model은 아니라고 했었죠. 
오늘 강의를 통해 Linear Classifier의 한계를 살펴보고, Multilayer Perceptron Model을 배우며 드디어 DL에 발을 들일 예정입니다!

<hr/>

## 1. Limitation of Linear Classifier

Linear Classifier의 한계는 그 이름이 함의하듯, Linearly Separable한 문제에만 적용이 가능하다는 것입니다. 
선형의 Decision Hyperplane으로 구분할 수 없는 문제에 대해서는, 좋은 Classification 결과를 줄 수 없습니다.<br/>

Linear Classifier의 한계를 간단한 예시들을 통해 살펴보겠습니다.<br/>

AND, OR, XOR Gate를 Linear Classifier로 구현할 것인데요, 여기서 AND, OR, XOR은 우리가 아는 논리연산자들입니다. 
앞으로 0은 False, 1은 True로 정의하겠습니다. 
또한, 여기서 사용될 Linear Classifier는 Binary Classification(이진 분류) 문제를 다루고 있으므로, 굳이 Label별 점수를 계산하여 argmax를 취해 분류할 필요가 없습니다.
단순히 하나의 Decison Hyperplane을 긋고, 그것을 기준으로 법선벡터의 양의 방향에 있는지, 음의 방향에 있는지를 통하여 Label들을 구분하면 됩니다.<br/>

함수로 표현하면, Decision Hyperplane의 법선벡터 W와 편향 b에 대해, Wx + b를 계산합니다. 그 결과는 Scalar 값이 나오겠죠? 이제 그 값을 h(x) = 1 if x > 0, 0 otherwise 라는 함수에 통과시켜서 분류하는 것입니다!<br/>

그냥 말로만 설명해서 와닿지 않을 수 있겠네요. 
예시를 보면, 더 잘 이해하실 수 있을 겁니다!

### A. AND Gate

이제, Linear Classifier로 AND Gate를 구현해 보겠습니다.<br/>

x0을 첫번째 Feature, x1을 두번째 Feature라 하면, AND Gate는 아래와 같이 작동해야 합니다.<br/>

| x0| x1| Result|
|:-:|:-:|:-----:|
| 1 | 1 | T |
| 1 | 0 | F |
| 0 | 1 | F |
| 0 | 0 | F |

![AND Plot](*)

이들을 구분하는 Decision Hyperplane 1개를 긋는 것이 목표입니다. 
Decision Hyperplane 위에 있는 점은 1로, 아래에 있는 점은 0으로 분류하는 것이죠.<br/>

![AND PLOT](*)

적절한 위치에 선을 그어 보면, x0 + x1 - 1.5 = 0이면 분류가 가능합니다.<br/>

숫자를 대입해 보면, 1 + 1 - 1.5= 0.5 > 0이므로 True로 분류되고, 나머지 점들은 비슷한 이유로 False로 분류가 된다는 것을 관찰할 수 있습니다.<br/>

Graph로 나타내면, 아래와 같습니다.<br/>

![AND Graph](*)

### B. OR Gate

이제, 비슷한 방법으로 OR Gate도 Linear Classifier로 구현해 볼까요?<br/>

| x0| x1| Result|
|:-:|:-:|:-----:|
| 1 | 1 | T |
| 1 | 0 | T |
| 0 | 1 | T |
| 0 | 0 | F |

![OR Plot](*)

Decison Hyperplane이 x0 + x1 - 0.5 = 0이면, 분류가 가능합니다.<br/>

Graph로 나타내면, 아래와 같습니다.<br/>

![OR Graph](*)

### C. XOR Gate

XOR Gate는 어떨까요?<br/>

그 전에, XOR에 대해 먼저 설명하겠습니다. 
XOR은 Exclusive OR로, OR을 만족하면서, 두 Input Feature가 서로 달라야 True인 논리연산자입니다. 
따라서, 아래와 같이 작동해야 합니다.<br/>

| x0| x1| Result|
|:-:|:-:|:-----:|
| 1 | 1 | F |
| 1 | 0 | T |
| 0 | 1 | T |
| 0 | 0 | F |

이제 이들을 평면 위에 나타내 봅시다.<br/>

![XOR Plot](*)

이들을 완벽히 분류하는 Decision Hyperplane을 그을 수 있나요?<br/>

아무리 생각해 봐도, 그러한 Decison Hyperplane을 찾을 수 없을 것입니다.<br/>

우리가 만든 Linear Classifier 코드를 이 상황에 맞추어 조금만 변형해 적용해 보면, AND Gate와 OR Gate는 100% 분류가 가능한 반면, XOR Gate는 75%만 분류 가능하다는 것을 확인하실 수 있습니다. 
(Weight를 2x1, Bias를 Scalar로 설정하고, Classification 방법도 바꿔야겠죠?)
비슷한 이유에서, 아래와 같은 원형의 데이터도 분류하지 못한다는 것을 알 수 있습니다.<br/>

![Circle Plot](*)

### D. Overcoming the Limitation

그렇다면, 이러한 문제들에 대해서는 분류를 어떻게 해야 할까요?<br/>

바로, 주어진 Feature를 변형하는 것입니다. 
조금 더 자세히 말하면, Feature를 다른 차원에 Mapping하여, Linearly Separable하게 만들어 주는 것이죠!<br/>

예시를 통해 더 알아봅시다. 
아까 XOR Gate를 다시 볼까요?<br/>

![XOR Plot](*)

이번에는, x0 + x1 - 1.5 = 0이라는 Hyperplane과, x0 + x1 - 0.5 = 0라는 Hyperplane 2개를 그어 줍니다. 
편의상, x0 + x1 - 105 = 0을 Classifier 1, x0 + x1 - 1.5 = 0을 Classifier 2라 부르겠습니다. 
또, h의 정의를 조금 수정하여, h(x) = 1 if x > 0, -1 otherwise라 하겠습니다. 
그러면 각각의 Classifier는 Input이 Hyperplane 위에 있으면 1, 아래에 있으면 -1을 결과로 줍니다.
이제, 각각의 점들을 Classifier 2개에 통과시키면, 새로운 좌표평면으로 결과가 Mapping됩니다.<br/>

![XOR Plot](*)

이때의 결과는 Linearly Separable합니다. 
따라서, 이제 하나의 Decision Hyperplane을 그어 주면 충분한 것이죠.

![XOR Plot](*)

Graph로 나타내면, 아래와 같습니다.<br/>

![XOR Graph](*)

수식으로 옮기면, h(x) = 1 if x > 0, -1 otherwise라 한다면, 우리의 분류 함수는, f = h(W2 * h(W1 * x + b1) + b2)입니다.<br/>

두개의 Linear Classifier가 중첩된 형태인데요, 이것을 우리는 **MLP(Multilayer Perceptron, 다층 퍼셉트론)**, 혹은 **ANN(Artificial Neural Network, 인공신경망)** 이라고 부릅니다.<br/>

그렇다면, 위에서 나온 Perceptron은 무엇일까요? 또, Neural이라는 말은 왜 붙는 것일까요?

<hr/>

## 2. Perceptron

### A. Perceptron as a Linear Classifier

AND Gate와 XOR Gate의 Graph를 다시 볼까요?<br/>

![AND Graph](*)
![XOR Graph](*)

Graph를 보면, Node 1개에 Input들이 들어옵니다. 
이때, 각각의 화살표에서 Input들에 가중치가 곱해집니다. 
이렇게 가중치가 곱해져서 들어온 Input들을 모두 더한 다음, h라는 분류 함수에 통과시킵니다. 
그리고, 그 결과가 다음 Node의 Input으로 들어가는 것이죠. 
지난 시간에 다룬 Computational Graph와 비슷하다고 생각하시면 됩니다. 
다만, W가 화살표로 표현되고, b가 Implicit하게 다뤄질 뿐인 것이죠!<br/>

이때, 네모친 영역들을 보면, 각각이 하나의 Decision Hyperplane을 표현하고 있음을 알 수 있습니다. 
하나하나가 Linear Classifier인 셈이죠. 
XOR Gate에서, Classifier 1, Classifier 2에 대해 먼저 분류를 하여 Linearly Separable한 좌표평면에 Mapping한 뒤, 다시 Classifier를 적용하여 분류했다는 것을 Graph를 통해 확인할 수 있습니다.<br/>

정리하면 각각의 영역에서 일어나는 일은,<br/>

1. Input들에 가중치를 곱한 값들을 받아서,<br/><br/>
2. 그들의 Linear Combination을 취한 뒤,<br/><br/>
3. 특정 함수(여기서는 h)를 통과시켜 다음 Node의 Input이 됨<br/>

입니다.<br/>

우리는 이러한 Graph의 한 부분인, Decision Hyperplane 1개를 긋는 Linear Classifier를 **Perceptron**이라고 부르겠습니다. 
그림으로 나타내면,<br/>

![Perceptron](*)

과 같습니다.

### B. Analogy to Neurons

이제 왜 Neural이라는 말이 나왔는지를 알아볼까요?<br/>

그러기 위해서는, Perceptron이라는 Model이 어떤 것을 모티프로 나온 것인지 살펴봐야 하는데요, Perceptron은 신경계의 단위인 **Neuron**을 모방해 만든 것입니다.<br/>

![Neuron](*)

위는 Neruon을 그린 그림입니다. 
Neuron의 작동 방식은 아래와 같습니다.<br/>

1. Dendrite로 신호를 수신하여, (자주 사용하는 Dendrite일수록 신호의 강도가 세짐)<br/><br/>
2. Cell Body에서 신호들을 합치고,<br/><br/>
3. 신호의 강도를 조절하여 Axon을 통해 신호를 내보냄<br/>

아까 살펴본 Perceptron의 작동 방식을 다시 볼까요?<br/>

1. Input들에 가중치를 곱한 값들을 받아서,<br/><br/>
2. 그들의 Linear Combination을 취한 뒤,<br/><br/>
3. 특정 함수(여기서는 h)를 통과시켜 다음 Node의 Input이 됨<br/>

Perceptron이 Neuron을 모방한 것이라는 것이 분명하죠?<br/>

이때, 3번에서 출력 신호의 강도를 결정하는 함수를 **Activation Function(활성함수)** 라고 부릅니다. 
대표적인 활성함수는 아래와 같습니다.

1. **Sigmoid**

![Sigmoid](*)

2. **Tanh**

![Tanh](*)

3. **RELU**

![RELU](*)

어떤 Activation Function을 사용하느냐에 따라 출력 강도가 달라지고, 그에 따라 분류 메커니즘도 달라질 것입니다. 
때문에, 이것도 하나의 Hyperparameter라 할 수 있겠죠?<br/>

이때, Activation Function이 없다고 생각해 볼까요? 
어떤 문제가 발생할까요?<br/>

![XOR Graph](*)

아까 XOR Gate의 Graph에서 Activation Function이 없다면, 전체 Graph를 Wx + b로 정리할 수 있을 것입니다. 
Activation Function을 제거했더니, 하나의 Linear Classifier가 된 셈이죠. 
Activation Function이 있어야, 다른 차원으로의 Mapping한 후에, Classification을 하는 것이 가능한 것입니다! 
비선형 문제도 분류할 수 있는 능력을 주는 것이죠!

<hr/>

## 3. MLP(Multilayer Perceptron)

### A. Idea

아까 XOR Gate의 Graph를 **MLP**라고 부른다고 했었죠? 
그 이름의 유래를 알아보겠습니다.<br/>

Node 1개에 Input들이 들어와서, 그들의 합이 Activation Function을 통과해 출력되는 Perceptron에 대해 배웠습니다. 
그리고 이것은 신경계의 Neuron을 모방한 것이라고 했습니다.<br/>

신경계의 Neuron은 아래와 같이 모여서 신경계를 형성합니다. 
Neuron들이 서로 연결되어 있는 모양이죠.

![Neural Network](*)

이러한 인간 신경계를 모방하고자, Neuron을 모방한 Perceptron을 만들었고, 이들을 연결하여 만든 것이 바로 **ANN**입니다. 
또한, Perceptron들이 여러 층으로 쌓여 있다고 해서 **MLP**라고도 부릅니다.

### B. Intro to DL

드디어 DL에 발을 들였습니다!<br/>

![AI vs ML vs DL](*)

저번에 AI, ML, DL을 구분하며, DL은 더 복잡한 Dataset을 더 복잡한 Model로 다룬다고 했습니다. 
그리고, 복잡한 Model은 인간의 신경계를 모방하여 구성하였다고 했습니다.<br/>

여기서 인간의 신경계를 모방한 Model이 바로 ANN입니다. 
이제서야 DL을 배운다고 말할 수 있는 것이죠!

### C. Hidden Layer

아까 XOR Gate의 Graph를 다시 봅시다.<br/>

![XOR Graph](*)

Perceptron들이 층으로 쌓여 있는, MLP입니다. 
Input이 들어오는 층을 **Input Layer**, 최종 Output이 나오는 층을 **Output Layer**라 합니다. 
그 사이에 있는 한 층은 **Hidden Layer**라고 부릅니다. 
이 HiddenLayer는 하나일 수도 있고, 여러 개가 될 수도 있습니다.<br/>

Hidden Layer들이 하는 역할은 무엇인가요?<br/>

아까 XOR Gate의 문제를 해결하면서 봤었죠! 

Hidden Layer를 통해, 이전 층의 Feature들을 다른 차원으로 Mapping합니다. 
이것을 통해, 원래의 Input Feature가 Linearly Separable하게 만들어 주는 것이죠! 
이때, Hidden Layer의 수가 많을수록 그 Mapping 과정이 복잡해지는 것이 되겠죠!<br/>

![Hidden Layer](*)

위 표에 Hidden Layer들의 효과가 잘 정리되어 있는데요, Hidden Layer 여러 개를 쌓으면 Input Dimension을 여러 개의 임의의 공간으로 나눌 수 있게 됩니다.<br/>

이제, MLP를 통해 Nonlinear Classification Problem을 풀 수 있게 되었습니다!<br/>

하지만, MLP를 학습시키면 모든 Nonlinear Classification Problem을 풀 수 있다는 것을 보장할 수 있을까요? 
MLP도 풀지 못하는 Nonlinear Classification Problem이 있다면, MLP는 Linear Classifier처럼 버려지지 않을까요?<br/>

### D. The Universal Approximation Theorem

MLP를 통해 Nonlinear Classification Problem들을 풀 수 있다는 것은 이미 증명되어 있습니다. 
그것을 **Universal Approximation Theorem**이라고 합니다. 
이 정리의 내용은,

```
“1개의 Hidden Layer를 가진 Neural Network를 이용해 어떠한 함수도 근사시킬 수 있다.”
(단, Activation Function이 Nonlinear하고, Hyperparameter들이 적절할 때)
```

입니다.<br/>

여기서 함수는, Classification 문제에서는 분류 함수, 즉 Decision Hyperplane들의 조합을 말하겠죠?<br/>

어떻게 이런 일이 가능한지, 정리의 증명을 아주 간략하게 살펴보겠습니다.<br/>

![UAT Pf](*)

위처럼 아주 복잡한 함수를 근사하려면, 함수를 여러 조각들로 쪼갠 후, 그 조각들을 각각 근사하는 것이 편하겠죠? 
각각의 조각은 하나의 Tower로 볼 수 있습니다.<br/>

![UAT Pf](*)

Activaton Function으로 Sigmoid를 쓴다고 해봅시다. 
Sigmoid Function을 s(x)라고 합시다. 
그러면, Wx + b가 s에 들어간, s(Wx + b)는 s(x)를 평행이동하고, 수축한 형태의 그래프일 것입니다. 
이런 식으로 Wx + b 연산으로 s(x)의 형태를 조금씩 조정해줄 수 있습니다. 

![UAT Pf](*)

두 함수의 위치를 위처럼 조정한 후에 빼주면, 위처럼 하나의 Tower가 나옵니다. 
이러한 Tower들을 합쳐서 복잡한 함수를 근사해 내는 것이죠!

따라서, MLP를 사용하면, Hyperparameter가 적절히 선택되었다면, Nonlinear Classification Problem들도 문제없이 풀 수 있습니다.

[[Illustrative Proof of Universal Approximation Theorem](https://hackernoon.com/illustrative-proof-of-universal-approximation-theorem-5845c02822f6)]에서 더 자세한 설명을 볼 수 있습니다!

### E. Looking Back at Backpropagation

MLP에 Backpropagation이 어떻게 적용되는지를 볼까요?<br/>

![MLP](*)

MLP의 Graph는 Computational Graph와 거의 비슷합니다. 
Weight가 화살표로 표현되고, Bias가 Implicit하게 다뤄진다는 것에 차이가 있죠.<br/>

하지만, 중심이 되는 아이디어는 같습니다.
MLP의 그래프 또한 복잡한 함수를 간단한 함수들의 합성으로 표현하고 있으며, Node간의 Chain Rule로 Gradient를 구할 수 있기 때문입니다.
그래서 Forward 방향으로는 함숫값이 계산이 되고, Backward 방향으로는 Gradient가 계산됩니다.<br/>

Linear Classifier의 Gradient를 구하는 과정은 간단합니다. 
따라서, 지난 시간처럼 아예 그 도함수를 바로 유도할 수도 있었죠. 
하지만, 위 그림과 같은 복잡한 MLP의 도함수를 손으로 구하기는 쉽지 않겠죠?<br/>

이때, Backpropagation을 통해 편미분 값을 계산하면 되는 것입니다! 
Backpropagation이 얼마나 강력한 아이디어인지 더 와닿으시나요?<br/>

### F. Implementation on Image Classification

MLP를 Image Classification에 적용하려면 어떻게 해야 할까요?<br/>

사진은 Matrix, 혹은 Tensor의 형태인데, 이것을 MLP의 Input으로 어떻게 넣을 수 있을까요?<br/>

해답은 단순합니다. 
Matrix나 Tensor를 Vector 하나로 편 다음, MLP의 Input으로 주는 것입니다. 
보통은 Column 단위로 Matrix를 뜯어서, 하나의 Vector로 합칩니다.<br/>

![Flatten](*)

오늘 다룰 MNIST Dataset은 손글씨에 대한 Dataset입니다. 
0 ~ 9를 손글씨로 쓴 흑백 사진과, 그 정답이 함께 붙어 있는 Dataset입니다. 
사진 하나는 28 * 28 크기입니다. 따라서, 이것을 784차원의 Vector로 펴서 MLP에 넣으면 되겠죠?<br/>

이때, MLP는 Tensorflow의 **Keras** 라이브러리로 만들 예정입니다. 
라이브러리가 굉장히 잘 짜여 있어서, 코딩보다는 레고 조각 맞추기에 가까울 것 같습니다. 
걱정 안하셔도 될 거에요!

<hr/>

## 4. Limitations of MLP

MLP Model의 아이디어와 Backpropagation의 아이디어가 제시된 것은 1980년대였습니다.<br/>

하지만, 이들이 빛을 본 것은 최근에 들어서였죠? 
어떤 문제들이 있어서 MLP가 실제로 사용되기까지 오랜 시간이 걸렸을까요?<br/>

1. Needs to Much Labeled Data for Training<br/>

말그대로, Training을 위해서는 엄청나게 큰 Dataset이 필요합니다. 
32 * 32 크기의, Label이 10개짜리인 Dataset도 60000개가 있어야 학습이 가능하기 때문이죠. 
당시에는 이렇게 큰 Dataset이 준비되지 않았습니다.<br/>

2. Vanishing Gradient<br/>

MLP의 Hidden Layer가 많아지면서, ANN이 깊어질 때 발생하는 문제입니다. 
Chain Rule을 통해 Gradient를 계산하다 보면, 너무 작은 값이 Upstream Gradient에 곱해져서 Gradient가 0이 되는 문제가 생깁니다. 
그러면, 그 이전의 Node들에 대해서는 학습이 제대로 이뤄지지 않겠죠?<br/>

3. Overfitting<br/>

Regularization을 통해 Overfitting을 어느 정도 방지하기는 했지만, 완전히 방지한 것은 아닙니다.<br/>

Training Loss는 분명 작았으나, Validation Loss는 엄청나게 큰 경우들이 생깁니다. 
이러한 Overfitting을 방지하기 위한 다양한 방법들이 필요합니다.<br/>

4. Gets Stuck in Saddle Point / Local Minima<br/>

현재 Gradient Descent 방식에 문제가 있다는 것을 눈치채셨나요?<br/>

우리의 목표는 Loss의 Global Mimimum으로 가는 것입니다. 
우리는 그것을 단순히, Gradient가 0이 되는 곳으로 정의했습니다.
Gradient가 0이 되면 더이상 Update가 되지 않기 때문이죠.<br/>

하지만, Gradient가 0이 되는 지점은 Global Mimimum 말고도 여러 지점이 있습니다. 
Local Mimima에서도 Gradient가 0이고, Saddle Point에서도 Gradient가 0입니다. 
따라서, Loss의 Local Mimima 혹은 Saddle Point에 갇힐 수도 있는 것이죠. 
이러한 문제들을 해결해야 합니다.<br/>

<hr/>

## Preview on Next Lecture(s)

이러한 문제들이 해결되면서 MLP와 NN(Neural Network)이 다시 주목받게 되었습니다.<br/>

비전처리 분야에서는, **CNN(Convolutional Neural Network)** 이라는 Model로 CIFAR-10 Dataset에 대해 90%대의 Accuracy를 기록했습니다. 
다음 수업에서는, 비전처리에 사용되는 NN Model인 CNN을 살펴보겠습니다.<br/>

그리고 6회차부터는 위에 제시된 MLP의 문제들이 어떻게 해결되었는가를 다루겠습니다.
