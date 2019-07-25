# Lec03_Model Optimization

## Contents

1. Optimization
2. Backpropagation

<hr/>

## From Last Lecture,

첫 번째 강의에서 **Model Optimization**을 하기 위해서는, 두 가지 질문들에 대한 답을 달아야 한다고 했습니다. 
많이 들어서 지겨우신가요?<br/>

1. Parameter가 얼마나 부정확한지 표현할 수치적 척도 필요<br/>
2. Parameter를 실제로 Update해주는 Algorithm 필요<br/>

1번 질문에 대한 답은, 지난 강의에서 다뤘습니다.<br/>

Parameter가 부정확한 정도를 **Loss**라는 숫자로 나타내기로 했습니다. 
그리고, Training Data에 대한 예측 결과와 그 정답을 비교하는 과정을 통해 Loss를 계산하는데, 이때 **Loss Function**이 개입된다고 했죠! 
또, Loss Function을 어떻게 정의하느냐에 따라 우리의 Model, 혹은 Decision Hyperplane의 Behavior가 결정된다고 했습니다. 
**Multiclass SVM Loss**와 **Cross-Entropy Loss**에 대해서 배웠고, 각각이 정의하는 좋은 Model이 무엇인지를 간략히 살펴봤습니다.<br/>

Model Optimization의 과정은 다음과 같다고 했는데요,<br/>

1. Training Set의 Data들을 Linear Classifier에 통과시켜서, 그 결과들을 정답과 비교.<br/>
2. 정답과 많이 다르다면, Parameter들의 값을 많이 바꿔줌. 정답과 차이가 거의 없다면, Parameter의 값을 조금씩만 바꿔줌.<br/>
3. 다시 1번으로.<br/>

우리는 지난 시간에 그 첫 과정을 끝마친 겁니다!<br/>

그러면 이제, 계산된 Loss를 바탕으로 Parameter들을 실제로 Update시켜주어야 합니다. 
Loss가 크다면, Parameter의 값들을 많이 조정해줘야 할 것이고, Loss가 작다면, Parameter의 값들을 조금만 바꿔 주면 될 것입니다.
이때, Parameter의 값을 얼마나, 그리고 어떻게 바꿔주는지를 오늘 수업을 통해 알아보겠습니다!

<hr/>

## 1. Optimization

### A. Idea of Gradient Descent

Parameter W와 b의 값을 어떻게 바꿔주어야 할까요?<br/>

가장 먼저 생각나는 방법은, 랜덤하게 바꿔주는 것입니다. 랜덤하게 Parameter W, b의 값을 바꿔 주며, Loss가 가장 작을 때의 W와 b를 채택하는 거죠!<br/>

당연히 좋은 방법은 아닙니다. 적절한 Parameter를 찾는다는 보장이 거의 없기 때문입니다.<br/>

다음에 취할 수 있는 방법은, 수학적인 분석을 적용하는 것입니다.<br/>

Loss를 계산하는 방법을 다시 볼까요?<br/>

![Loss](*)

이 수식은 결국 Input으로 들어온 Dataset {(x, y})와 Parameter W, b에 대해, Loss라는 Scalar Output을 계산하는 것입니다. 
그러면, W와 b에 대한 함수라고 해석할 수 있는거죠.<br/>

이제, Loss를 Visualize해봅시다. 다양한 W, b의 조합에 따른 Loss를 다차원 공간상에 나타낼 수 있겠죠?

![Loss Graph](*)

우리의 목표는 Loss를 최소화하는 것입니다. 
따라서, 위에서 그린 그래프가 최솟값을 가질 때의 W와 b값을 얻고 싶은 것이죠. 
현재의 Loss 값에서, Loss의 Global Minimum으로 가려면 어떻게 해야 할까요?<br/>

W가 행렬, b가 벡터여서 상상이 잘 안될 것 같네요. 
그러면, W와 b가 모두 Scalar라고 단순화하여 생각해 봅시다. 
여기서 더 단순화해서, 적절한 b는 이미 찾았다고 가정하고, W에 대한 L의 그래프를 그려 봅시다.<br/>

![Loss Graph_Simple](*)

현재 위치에서 Loss의 Gloabal Minimum으로 굴러 떨어지고자 하는 것이 우리의 목표입니다.<br/>

여기서는, 현재 위치에서 계산한 그래프의 기울기 방향을 보고 굴러 떨어지면 되겠죠? 
기울기, 혹은 미분계수가 양수가 나오면, 음의 방향으로 W를 Update시켜 주면 L의 크기가 작아집니다. 
반대로, 기울기가 음수가 나오면, 양의 방향으로 W를 Update시켜 주면 L의 크기가 작아집니다. 
수식으로 나타내면,<br/>

![Gradient Descent](*)

입니다.<br/>

-dL / dW이 Update시켜줄 방향을 알려 주고, 그 크기에 lr만큼을 곱해 실제로 Update되는 정도를 계산합니다. 
이때, lr는, Learning Rate라 부르는 Hyperparameter입니다.<br/>

위에서는, 적절한 b를 이미 찾았다 가정했습니다.
하지만, 그럴 리는 없죠?
우리는 Parameter W, b 둘다 적절한 값을 찾아 주어야 합니다.
따라서 이제, b에 대해서도 생각해 봅시다.<br/>

이번에는, 3차원 공간상에 Loss 그래프가 그려질 것입니다. 
이때의 Update 방향은 어떻게 찾을 수 있을까요?<br/>

각각의 변수에 대해 편미분을 해 주면 되겠죠? 
그러면 - dL / dW는, b가 Constant라 가정하고, L이 작아지는 W의 Update 방향을 나타낼 것이고, - dL / db는, W가 Constant라 가정하고, L이 작아지는 b의 Update 방향을 나타낼 것입니다. 
그 다음, W와 b를 각자의 편미분 결과를 가지고 Update시켜주면 되겠습니다.<br/>

이러한 과정을 Gradient의 방향으로 Parameter를 Update시켜주어 Loss 그래프를 따라 내려간다 하여, **Gradient Descent(경사하강법)**라고 부릅니다.<br/>

사실, Parameter W와 b는 Scalar가 아니죠? 
지금까지는 이해의 편의를 위해 Scalar라고 가정했을 뿐입니다. 
실제 W는 행렬이고, b는 벡터였습니다. 
하지만, 행렬과 벡터의 Entry 하나하나를 뜯어서 보면, 아까와 동일하게 생각할 수 있습니다.<br/>

먼저, W를 어떻게 Update하는가를 볼까요?<br/>

Iris Classification에서, W는 3 x 4 행렬이었습니다. 
Entry가 총 12개 있는 행렬이죠. 
그러면, W의 Entry 12개에 대해서 그려진, Loss 그래프를 상상하실 수 있으신가요? 
여기서도, 각각의 Entry에 대해, L을 편미분하여 각각을 Update시켜 주면 되겠습니다. 
모든 Entry를 Update시키는 과정을,<br/>

![Gradient Descent](*)

라 정리하여 표현할 수 있는데요, dL / dW_old는, L을 W의 Entry들로 각각 편미분한 결과가 들어가는 행렬입니다. 
이러한 행렬을 **Gradient**라 하겠습니다.<br/>

벡터인 b도 마찬가지 방법으로, Elementwise하게 Gradient dL / db를 구하여, Update시켜주면 됩니다.<br/>

이제 Parameter를 Update시키는 방법과 그 수식을 배웠으니, Gradient만 구해주면 되겠죠?<br/>

하지만, Gradient를 구하는 것은 쉬운 일은 아닙니다. 
Loss는 꽤나 복잡한 수식으로 계산한 결과이기 때문입니다. 이를 어떻게 편미분할 수 있을까요?

### B. Calculating the Gradient

> **Numeric Gradient**

첫번째 Approach는, 미분의 정의를 활용하는 것입니다.<br/>

h가 0에 가까워질 때,<br/>

![Derivative Def](*)

의 값을 미분계수라고 합니다.<br/>

따라서, 이 h에 0.00001같은 작은 수를 대입해 값을 계산해 냅니다. 
이렇게 얻은 편미분 값들로 얻은 Gradient를, **Numeric Gradient**라고 합니다.<br/>

딱 봐도 좋은 방법은 아닙니다.<br/>

우선, Gradient의 근사값만 얻을 수 있습니다. 
h에 작은 수를 대입해서 구한 것이지, h가 0에 접근할 때의 값을 구한 것은 아니기 때문입니다.<br/>

또한, Gradient를 구하는 방법이 효율적이지 못합니다. 
Parameter의 값 하나하나마다 h를 더하고, Loss를 구하고, 그것들에 대해 또 계산을 해줘야 하기 때문입니다. 
Iris Classification의 경우, W 행렬의 12개 값과, b 벡터의 3개의 값에 대해 하나하나 편미분 값의 근사치를 구해야 합니다. 
한번 Parameter를 Update하기 위해, 15번의 복잡한 Loss 계산들을 해야 하는 것이죠.<br/>

그렇기 때문에, Numeric Gradient는 Update에 사용하지 않습니다.

> **Analytic Gradient**

미분계수를 구하는 방법은 미분의 정의를 활용하는 방법도 있지만, 도함수에 값을 대입하는 방법도 있죠? 
이번에는 도함수를 통해서 Gradient를 구해 보겠습니다. 
이렇게 구한 Gradient는 **Analytic Gradient**라 합니다.<br/>

그런데, Loss를 구하는 복잡한 과정을 어떻게 편미분할 수 있을까요?<br/>

지금 Loss의 도함수는 구하기 조금 귀찮을 뿐이지, 구할 수는 있을것 같다고요? 
그럴 수도 있겠네요. 
하지만, 우리가 지금 다루고 있는 Linear Classifier는 아주 단순한 Model이라는 것을 상기해야 합니다. 
앞으로 다룰 아래와 같은 Model들은 굉장히 복잡한 연산을 통해 Classification을 합니다.<br/>

![MLP Architecture](*)

이러한 Model의 Loss의 도함수를 손으로 계산하는 것은 거의 불가능에 가까울 것입니다. 
잠시 후, 이러한 복잡한 함수의 도함수 값을 계산하는 방법을 다루겠습니다.<br/>

실제 Model Optimization에서는, Numeric Gradient가 아닌, Analytic Gradient를 계산합니다.<br/>

그렇다고 Numeric Gradient의 쓸모가 아예 없는 것은 아닙니다. 
Analytic Gradient를 구할 때, 도함수를 통해 구한 미분계수가 정확하다는 것을 어떻게 알 수 있을까요?<br/>

이때, Numeric Gradient를 사용합니다. 
Analytic Gradient의 값과 Numeric Gradient의 값이 일치하거나, 거의 비슷하다면, 도함수를 옳게 구했다는 것을 알 수 있습니다. 
Numeric Gradient로는 검산을 하는 것이죠!<br/>

또한, 아무리 Analytic Gradient를 사용한다고 해도, 계산량은 여전히 많습니다. 
Training Set의 크기가 60000이라면, Training의 Iteration마다 60000개의 Input에 대해 계산한 Loss를 편미분해줘야 할 것입니다.<br/>

이러한 계산량을 줄이기 위해, 우리는 Training의 매 Iteration에서 Training Set에서 일부를 Sampling해서 Mini Batch를 구성합니다. 
그리고, 그 Batch에 대한 Loss를 계산하여 Parameter를 Update시켜 줍니다. 
Training Set 전부를 사용하지 않고, 그 일부인 Mini Batch를 이용하여 전체 Loss를 근사하고 있으므로, 이것을 **SGD(Stochastic Gradient Descent, 확률적 경사하강법)** 이라고 부릅니다.

<hr/>

## 2. Backpropagation

Loss의 계산식을 전개하면 아주 복잡한 함수가 됨을 확인했습니다. 그 도함수를 구하기는 쉽지 않겠죠. 
그 도함수를 구하는 방법이 바로 **Backpropagation**입니다.

Backpropagation을 간략히 설명하자면, 다음과 같습니다.<br/>

1. 복잡한 함수는 간단한 함수나 연산들의 합성함수로 나타낼 수 있다. 그것을 **Computational Graph**로 시각화할 수 있다.<br/>

2. **Chain Rule(합성함수 미분법)** 을 적용하여, 간단한 함수이나 연산들의 도함수들로 원래 함수의 도함수를 구해 낸다. <br/>

고등학교 수학으로 돌아가, exp(cosx)를 어떻게 미분하는지를 생각해 볼까요?<br/>

f(x) = exp(x), g(x) = cosx, h(x) = f(g(x))라고 합시다. 
그러면, h’(x) = g’(x) * f’(g(x)) = - sinx * exp(cosx)와 같이 합성함수 미분법을 적용할 수 있었죠? 
dh / dx = dh / dg * dg / dx를 계산한 것인데요, 이것을 **Chain Rule**이라고 합니다.<br/>

복잡한 함수를 간단한 함수들로 쪼개면, 그 관계들을 알아야 Chain Rule을 적용할 수 있습니다. 
그 관계를 알아보기 쉽게 시각화한 것이 바로 **Computational Graph**입니다. 
Computational Graph에서, 각 Node는 Operator이고, 각 Leaf는 Operand가 됩니다. 
Node 1개에 Input들이 들어와, 간단한 함수를 통과하여 출력이 계산되는 형태이죠.<br/>

예시를 통해 더 살펴보도록 하죠!

### A. Scalar

Input들이 모두 Scalar일 때를 먼저 볼까요?<br/>

f(x, y, x) = (x + y) * z라는 함수를 봅시다. 우리의 목표는, (x, y, z) = (1, 2, 3)에서의 f의 Gradient를 구하는 것입니다. f를 (1, 2, 3)에서
각 변수에 대하여 편미분한 값들을 구하는 것이죠.<br/>

중간 계산 결과를 q = x + y라 하면, f = q * z가 됩니다.<br/>

먼저, f의 계산 결과를 구하려면, (1, 2, 3)을 Graph에 순방향으로 통과시키면 됩니다. 
q = 1 + 2 = 3이므로, f = 3 * 3 = 9가 됩니다.<br/>

다음, Gradient를 구해 봅시다. 
df / dz = q = 3입니다. 
또, df / dq = z = 3입니다. 
이제, q = x + y이므로, 이를 이용하여 df / dx, df / dy를 구해 주면 됩니다. 
그러면, df / dx = df / dq * dq / dx = 3 * 1 = 3이고, df / dy는 마찬가지 방법으로, 3이 됩니다.<br/>

Gradient를 구한 방법을 다시 보면, 전체 함수의 편미분값을 간단한 함수들의 편미분값의 곱으로 구하는, Chain Rule이 적용되었음을 알 수 있습니다.<br/>

이제, Node 하나에 초점을 맞춰서 관찰해 봅시다.<br/>

![Node](*)

중간 Node에 x, y라는 input이 들어와서, q = x + y가 출력됩니다. 
그리고, 그 값은 다시 다음 Node의 input으로 들어갑니다. 
이를 **Forward** 방향이라고 합니다. 
함수의 값을 계산할 때는, Computational Graph에서 Forward 방향으로 계산이 수행되는 것이죠.<br/>

Gradient를 구할 때는 어떤가요?<br/>

앞의 Node로부터, 미분값 df / dq = z가 전달됩니다. 
이제, 현재 Node를 미분한 dq / dx, dq / dy를 곱해 주어, 이전 Node로 그 값을 전달합니다. 
아까와는 반대 방향으로 계산이 진행되죠? 
이 방향을 **Backward**이라고 부릅니다. 
위의 Node로부터 도착한 **Upstream Gradient**가, 현재 Node에서의 **Local Gradient**와 곱해져, 이전 Node로 전달되는 것이죠.<br/>

아까 Gradient를 구하기 위해 Backpropagation을 진행한다고 했었죠? 
Backpropagation의 Back이 Computational Graph에서의 계산 방향을 의미하는 것입니다. 
최종 계산 결과로부터 Input들로 미분값이 전파된다는 뜻입니다!<br/>

이제, f가 최소화되는 방향으로 가려면, - Gradient = - (3, 3, 3) 방향으로 (x, y, z)가 이동하면 되겠죠?<br/>

위에서 다룬 함수는 Backpropagation, Computational Graph, Chain Rule이 무엇인지를 알아보기 위한 아주 간단한 예시였습니다.
사실, 위의 예시는 도함수를 직접 손으로 구하는 것이 훨씬 빠르죠. 
그래서, 더 복잡한 함수를 통해 Backpropagation이 얼마나 강력한 Idea인지 알아보겠습니다.<br/>

f = 1 / (1 + exp(- w0 * x0 - w1 * x1 + w2)라고 합시다. 
우리의 목표는, Gradient df / dw = (df / dw0, df / dw1, df / dw2)를 구하는 것입니다.<br/>

![Sigmoid](*)

이제, Backpropagation을 통해 복잡한 함수의 미분계수를 계산할 수 있다는 것이 이해되시나요?<br/>

복잡한 함수를 간단한 함수들로 쪼개어 그것을 Computational Graph로 시각화한 후, Chain Rule을 통해 미분계수를 계산해낸 것입니다!

** Input 값 1개가 여러가지 함수들의 Input으로 들어간다면 어떻게 되나요? <br/>

각각의 함수에서, Input의 Gradient를 구해 줍니다. 그 다음, Gradient들을 더해 준 것이 최종적으로 Input에 대한 Gradient가 됩니다.

### B. Vector

위의 예시들은 Input으로 Scalar 값들이 들어갔습니다. 
하지만, 우리가 다루는 Model은 Input이 행렬, 혹은 벡터이죠? 
따라서, 벡터에 대해서도 Backpropagation을 적용하여 편미분값을 구하는 방법을 알아야 합니다.
벡터가 Input으로 들어가고, Output으로 다시 벡터가 나오는 Node를 볼까요?<br/>

이때, Scalar인 L을 벡터인 z로 편미분한다는 것은 무슨 뜻일까요? 
또, z와 y는 벡터인데, z를 y로 편미분한다는 것은 무슨 뜻일까요?<br/>

먼저, dL / dz는, L을 z의 각 Entry로 편미분한 값들이 들어있는 벡터입니다. 
이제, Chain Rule을 적용하여 dL / dy를 구하려면, Upstream Gradient인 dL / dz에 dz / dy를 곱해 주어야 합니다. 
dz / dy는, z의 각 Entry를 또다시 y의 각 Entry로 편미분한 값들이 들어있는 행렬입니다. 
이를 **Jacobian Matrix**라고 합니다.
예를 들어 볼까요?<br/>

![Jacobian Example](*)

Input x가 max 함수를 거쳐, Output y가 출력되는 Node입니다. 
여기서 Jacobian Matrix는 y의 각 Entry에 x의 각 Entry가 얼마나 영향을 미치는지를 표현하고 있는 것이죠!

### C. Matrix

그러면, 행렬이 Input으로 들어가고, Output도 행렬인 Node는 어떨까요?<br/>

마찬가지로, Output z의 Entry들을 Input y의 Entry들로 편미분해 주어야겠죠? 
이것을 **General Jacobian**이라고 합니다. 
그런데, 이 General Jacobian이 어떻게 생겼을지 상상이 가시나요? 
아까 Jacobian은 행렬이었는데, 지금은 행렬을 행렬로 편미분하고 있으므로, 정보량이 더 많습니다. 
그냥 행렬로 표시하기는 어려워진 것이죠.<br/>

이러한 고차원 Matrix를 **Tensor**라고 부릅니다. 
General Jacobian은 Tensor가 되는 것이죠. 
시각화를 한다면, 직육면체를 생각하시면 될 것 같습니다. 
z00을 y의 Entry들로 편미분한 것이 하나의 행렬이 되고, 그것을 확장하여 z의 첫 행을 y의 Entry들로 편미분해준 것은, 행렬들을 위로 쌓은 Cube가 되는 것이죠. 
그것을 다시, z의 모든 행으로 확장한다면, Cube들을 쌓은 하나의 직육면체가 될 것입니다.<br/>

사실, 이러한 Tensor가 어떻게 생겼는지 이해하는 것은 지금 단계에서는 중요하지 않습니다. 
따라서, 그냥 ‘정보량이 많은 General Jacobian이 필요하구나’ 정도만 이해하셔도 충분합니다.<br/>

이때, 이 General Jacobian의 정보량이 너무 많다는 것이 문제가 됩니다. 
앞으로 다룰 Feature들은 아주 큰 Dimension을 가질 것입니다. 
CIFAR-100의 경우, Input Feature가 3072개, Label이 100개이니까요. 
Input y가 64x4096, Output z가 64x4096이라 한다면, General Jacobian의 크기는 256GB가 될 것입니다. 
이렇게 큰 General Jacobian을 계산하는 것은 불가능합니다. 
Node 한번의 계산을 위해 256GB만큼의 정보를 저장하고 있는 것은 아주 비효율적이기 때문입니다.<br/>

따라서, 우리는 General Jacobian을 사용하지 않을 것입니다. 
구하고자 하는 dL / dy는 하나의 행렬입니다. 
그래서 General Jacobian을 통하지 않고, Elementwise하게 dL / dy를 계산할 것입니다.
가장 간단한 연산인 곱셈을 예시로 들어 보겠습니다.<br/>

![Matrix Multiplication Derivative](*)

Upstream Gradient는, dL / dy입니다. 
이제, Chain Rule을 통해 dL / dW를 구하고자 합니다.
General Jacobian을 사용하지 않고, Elementwise하게 dL / dW를 구한다고 했죠? 
그럼 먼저 dL / dW00에 Chain Rule을 적용해 봅시다.<br/>

![Matrix Multiplication Derivative Derivation](*)

이 되겠죠?<br/>

마찬가지 방법으로, 

![Matrix Multiplication Derivative Derivation](*)

이 됩니다.<br/>

(W00이라는 값이 곱셈 연산 두개에 들어가므로, 각각에 대해 미분값을 구한 후, 더해주는 것입니다!)

이것을 행렬로 정리하여 더 깔끔하게 나타내면, dL / dW = dL / dy * x’가 되겠습니다.<br/>

Gradient를 계산할 때, General Jacobian을 구하지 않고, Elementwise한 Chain Rule을 적용한 것이죠. 
여기서 Local Gradient는, dL / dW를 Elementwise하게 구하는 과정에 Implicit하게 포함되어 있는 것입니다!<br/>

** 위의 방법이 어떻게 더 효율적인가요?<br/>

dL / dW를 구하는 과정에서, General Jacobian Approach와 Elementwise한 Approach 모두 Local Gradient를 계산하고 있기는 합니다. 
General Jacobian을 구하려고 하니, 그 크기가 너무 커져서, Elementwise하게 Gradient를 계산한 것입니다.<br/>

General Jacobian이 있으면, 한번에 많은 숫자를 저장하고 있어야 하기 때문에 비효율적입니다.<br/>

Elementwise한 계산을 할 때는, 필요할 때마다 Local Gradient의 Entry를 계산하면 되니까, 저장 공간이 많이 필요하지는 않은 것이죠.<br/>

** 도함수에서 행렬의 Transpose(전치)가 나오는 기준이 무엇인가요?<br/>

dL / dW = dL / dy * x’는 그저 Elementwise하게 편미분한 결과를 정리해서 나타내기 위함이지, 행렬의 전치에 수학적인 이유가 숨어있는 것은 아닙니다.<br/>

** 수식이 이해가 잘 안돼요...!<br/>

바로 이해가 되지 않아도 괜찮습니다.<br/>

구글링을 하면, 행렬 연산들에 대한 Gradient들이 정리된 수식으로 나오므로, 그것을 코드로 옮기기만 하면 됩니다.<br/>

사실, 코드로 옮긴 것들도 이미 라이브러리로 구현되어 있기 때문에, 자세한 Detail은 이해가 덜 되어도 문제 없습니다.<br/>

실제로 우리가 뒤에 다룰 Tensorflow의 Keras 라이브러리에서는, Model의 형태를 정의한 후, Model.fit()만 해주면 Backpropagation이 자동으로 진행됩니다. 
Model이 복잡해도, Gradient를 알아서 잘 계산하고, Parameter를 그에 맞게 Update시킵니다.<br/>

그러니, 걱정하지 않으셔도 됩니다.<br/>

Matrix에 대한 Backpropagation을 자세히 정리해 놓은 자료는, [[Linear Backprop](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)]
 [[딥러닝 역전파 수식 행렬의 전치(Transpose) 기준?](http://taewan.kim/post/backpropagation_matrix_transpose/)] 입니다. 
최대한 Backpropagation을 쉽게 다루고자 Detail한 내용들을 많이 생략했는데요, 위 자료들에 오늘 한 이야기가 수학적으로 잘 정리되어 있습니다.

### D. Implementation on our Linear Classifier

우리가 지난 시간에 배운 Loss는 대략 아래같은 구조의 Computational Graph를 가질 것입니다.<br/>

![Loss Computational Graph_Simple](*)

이것을 W(와 b)에 대해 편미분하여 Gradient를 얻는 것이 목표입니다.<br/>

더 자세한 Computational Graph는,

![Loss Computational Graph_Detailed](*)

입니다.<br/>

이제, Backpropagation을 통해 도함수를 유도해 보겠습니다.<br/>

![Linear Classifier Gradient Derivation](*)

** Backpropagation은 편미분한 값들이 Node들 간에 전달되는 것인데, 이렇게 전체 도함수를 구하는 것은 Backpropagation이 아니지 않나요?<br/>

맞습니다. 
원래는, Node들 간에 편미분한 값들을 전달하여 미분값을 얻습니다.<br/>

다만, Linear Classifier는 단순한 Model이고, Chain Rule 이해를 돕기 위해 전체의 도함수를 유도하겠습니다. 
Backpropagation에서의 편미분 값 전달 단계를 숫자가 아닌 수식으로 표현했다고 생각하시면 될 것 같습니다.<br/>

따라서, 오늘 코딩실습에서도 Gradient Update를 구현하겠지만, Backpropagation은 구현하지 않을 예정입니다. 
지금 유도할 도함수에 값을 대입하여 Gradient 값을 얻고, 그것을 원래 Parameter에서 빼주어 Update시키는 과정을 구현할 것입니다.

![Gradient Derivation](*)

이제, 이렇게 유도한 도함수 식을 가지고, Linear Classifier의 Parameter를 Update시켜 봅시다!

<hr/>

## Review on Today’s Lecture

오늘은 Loss를 바탕으로 Parameter를 Update시켜주는 방법을 배웠습니다.<br/>

Loss에 대한 Parameter W, b의 Gradient를 계산해 주어, Loss 그래프의 Global Minimum으로 가는, Gradient Descent 방법으로 Parameter를 Update합니다.<br/>

이때, 복잡한 Loss 함수의 Gradient를 구하는 방법으로, Backpropagation을 배웠습니다. 
복잡한 함수를 단순한 함수들의 합성으로 표현한 후, 그것을 Computational Graph로 시각화했습니다. 
그 다음, Chain Rule을 활용해, 각 Node에서 Upstream Gradient에 Local Gradient를 곱해 주는 방식으로 Gradient를 실제로 구해 봤습니다.<br/>

다시 강조드리지만, Backpropagation이 바로 이해가 안되는 것이 정상입니다. 
저도 그 아이디어를 이해하기까지 한달이라는 시간이 걸렸고, 아직도 헷갈리니까요. 
충분한 시간을 두고 고민해 보시면 됩니다.<br/>

또한, 행렬이 개입된 Gradient를 구하는 것은, 구글링하여 찾은 수식을 코드로 옮기면 해결이 되는 문제입니다. 
그리고 이런 고민도 할 필요가 없는 것이, 이미 수많은 라이브러리가 Backpropagation을 알아서 해주기 때문입니다!

<hr/>

## Preview on Next Lecture

이로써, 우리의 두번째 Model을 완성했습니다. 
Linear Classifier를 만들고, Parameter들을 Update시켜 주었죠!<br/>

하지만, Linear Classifier는 이미지 분류에 적합한 Model은 아닙니다. 
이름에 그 태생적인 한계가 숨어 있는데요, Linearly Separable한 문제에 대해서만 분류가 가능하기 때문입니다.<br/>

![Nonlinear Classification Problem](*)

가령, 위처럼 원형으로 분포된 데이터는 Decision Hyperplane으로 구분할 수 없겠죠?<br/>

이런 문제는 Linear Classifier로는 해결할 수 없기 때문에, 새로운 Model을 배울 때가 된 것입니다.<br/>

미리 Preview를 하자면, Multilayer Perceptron Model이 나오고, ANN(Artificial Neural Network, 인공신경망)이 등장할 것입니다.
비로소 DL에 발을 들이는 것이죠!<br/>

다음 시간도 차근차근 설명드리도록 하겠습니다!
