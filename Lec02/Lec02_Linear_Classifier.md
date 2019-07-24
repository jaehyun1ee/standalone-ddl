# Lec02_Linear Classifier

## Contents

1. Loss
2. Loss Function
3. Regularization

<hr/>

## From Last Lecture,

분류 Model을 만들 때, **Data-Driven Approach**로, **Nearest Neighbor**와 **K-Nearest Neighbor Algorithm**을 적용해 보았습니다.<br/>

이들의 성능이 좋지 않다는 것을 확인한 후, 특정 변수에 Label별 특징을 값으로 저장하는, **Parametric Approach**를 생각해 냈습니다.<br/>

그리고 **Linear Classifier**가 무엇인지 배웠는데요, Linear Classifier는Parameter W(Weight, 가중치)와 b(Bias, 편향)을 통해 feature x를 Label별 점수 s로 Mapping합니다.
그 후, 점수의 최댓값으로 Prediction을 내립니다.<br/>

W와 b가 처음부터 분류를 잘하도록 설정이 된다면 좋겠지만, 이들 값은 프로그램이 학습을 통해 찾아야 하는 값들입니다. 
현재의 부정확한 Parameter를 Update시켜서, 분류를 잘 하는 Parameter로 만들어줘야 하는 것이죠.
이를 **Model Optimization**이라 부릅니다.<br/>

Model Optimization을 하기 위해, 우리는 두 가지 질문들을 답해야 합니다.<br/>

1. Parameter가 얼마나 부정확한지 확인할 수치적 척도인 Loss가 필요하다.<br/>

2. 1에서 구한 Loss를 바탕으로, Parameter를 실제로 Update하는 Algorithm이 필요하다.<br/>

오늘 수업에서는, 첫 번째 질문에 답을 달아 보겠습니다.

<hr/>

## Loss

Model Optimization은 다음과 같은 과정으로 진행됩니다.<br/>

1. Training Set의 Data들을 Linear Classifier에 통과시켜서, 그 결과들을 정답과 비교.<br/>

2. 정답과 많이 다르다면, Parameter들의 값을 많이 바꿔줌. 정답과 차이가 거의 없다면, Parameter의 값을 조금씩만 바꿔줌. <br/>

3. 다시 1번으로.<br/>

2번 과정에서 우리는 Parameter가 얼마나 부정확한지를 판단해야 합니다. 
이때, ‘많이’, 혹은 ‘적게’는 명확한 표현이 아닙니다. 
프로그램을 돌리려면, 이들 표현을 수치적으로 바꿔줄 필요가 있습니다.<br/>

Training을 할 때, Training Dataset에 대해, 현재 Parameter의 부정확한 정도를 **Loss over Dataset**이라 부르겠습니다.<br/>

학습을 통해, Parameter를 조정하여, Loss를 최소화하는 것이 우리의 목표가 되겠죠?<br/>

이때, Data 1개에 대해 Loss를 계산하는 함수를 **Loss Function**이라 부르겠습니다.

<hr/>

## Loss Function

Loss Function은 Linear Classifier를 통과해 나온 결과와, 정답을 비교하여 틀린 정도를 숫자로 반환해 줍니다.<br/>

이때, Loss Function을 어떻게 정의하느냐는, 좋은 Model을 어떻게 정의하느냐와 같은 질문입니다. 
우리는 Loss Function을 통해 Loss를 계산하여, 그것을 토대로 Model을 Optimize할 것이기 때문입니다.<br/>

가령, “좋은 Linear Classifier는, 정답 Label의 점수가 다른 Label의 점수들보다 100씩은 높아야 해.”라고 정의한다면, Loss Function을 아주 Sensitive하게 정의해야 할 것입니다. 
정답 Label의 점수가 다른 Label의 점수에 비해 월등히 높아야 하기 때문이죠.<br/>

Linear Classifier에 대한 기하적인 해석을 생각해 보면, Loss Function을 어떻게 정의하느냐에 따라, 우리가 원하는 최적의 Decision Hyperplane의 성질도 정해지는 것이 되겠죠!<br/>

Classification에서 자주 쓰이는 Loss Function에는 다음 2가지가 있습니다.<br/>

**1. Multiclass SVM Loss**<br/><br/>
**2. Cross-Entropy Loss**<br/>

이제 각각에 대해 알아보도록 하겠습니다.

### A. Multiclass SVM Loss

Multiclass SVM Loss를 가장 잘 설명한 내용입니다.<br/>

```
“The SVM Loss is set up so that the SVM wants the corret class for each input to have a score higher than the incorrect classes by some fixed value delta.”
```

다시 말하면, 정답 Label의 점수가 나머지 Label의 점수보다 Delta만큼 크기를 바라는 것이죠.<br/>

Delta가 10이라고 생각해봅시다. 
Training set의 Data (1, 3, 2, 4)를 Linear Classifier에 통과시켜 점수 (21, 28, 9)를 얻었습니다. 
그리고, Data의 Label은 1(index는 0부터 시작)입니다. 
점수의 argmax를 취하면, 1이 나오므로, 현재 Parameter가 분류를 알맞게 하고 있기는 합니다. 
하지만, 정답 Label의 점수인 28이 첫 Label의 점수인 21보다 7이 큽니다. 
우리가 Delta를 10으로 설정했으므로, 현재의 Model은 우리가 바라는 좋은 Model은 아닌 것이죠. 
Train을 거듭해서 더 적절한 Parameter를 찾아 주어야 합니다.<br/>

그러면, Multiclass SVM Loss를 계산하여 학습시킨 Linear Classifier는 기하적으로 어떤 성질을 가질까요?<br/>

Multiclass SVM Loss를 통해, 우리는 Max-Margin Property를 가지는 Decision Boundary를 찾고자 합니다. 
여기서, Max-Margin Property를 가진다는 말은, Decision Boundary가 두 Label을 General하게, 절반으로 나눠준다는 뜻입니다. 
Decision Boundary에서 법선벡터의 방향으로 더 멀어질수록 해당 Label에 대한 점수가 높아진다고 했는데요, Max-Margin Property를 가진다는 것은, 정답 점수들을 높이는 Decision Hyperplane을 그린다는 것입니다. 그림으로 보면 이해가 더 쉬울 것 같네요.<br/>

Multiclass SVM Loss가 어떻게 Max-Margin Property로 이어지는지는 다루지 않겠습니다. 
이에 대한 설명은 [[Support Vector Machines](cs229.stanford.edu/notes/cs229-notes3.pdf)]에서 찾아보실 수 있는데요, 최적화(OR)에 대한 배경지식이 있으시면 이해하실 수 있을 것 같습니다. 
저도 아직 완벽히 이해하지는 못했지만, 아이디어 정도는 이해할 수 있었습니다.

### B. Cross-Entropy Loss

Cross-Entropy Loss는 앞서 설명한 Multiclass SVM Loss와는 다른 Approach를 취합니다. 
이제, 점수에 대한 확률적인 해석이 들어가게 됩니다.<br/>

우선, **Softmax Function**을 통해, Label별 점수들을 Label별 확률로 Mapping시켜 줍니다. 
먼저, Exp 함수를 통해 점수들을 Normalize시켜준 후, Normailzed Sum으로 각 값들을 나눠 주어 [0, 1]의 값들로 바꿔 줍니다. 
이들 숫자는 이제 각 Label에 대한 조건부 확률로 해석할 수 있습니다. 
모든 확률들을 다 더하면 1이 되는 것도 확인할 수 있습니다.<br/>

아까와 마찬가지로, Softmax Function을 통과한 결과가 왜 확률로 해석되는지는 다루지 않겠습니다. 
[[Multinomial Logistic Regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)] [[Logistic, cross-entropy loss의 확률론적 의미](https://de-novo.org/2018/05/03/logistic-cross-entropy-loss%EC%9D%98-%ED%99%95%EB%A5%A0%EB%A1%A0%EC%A0%81-%EC%9D%98%EB%AF%B8/)] 문서를 참고하시면 더 자세한 설명을 보실 수 있습니다.<ㅠㄱ/>

이제, 이렇게 조건부 확률로 해석한 점수를 가지고, **MLE(Maximum Likelihood Estimation, 최대우도추정)**을 하는 것이 Cross-Entropy Loss의 목표입니다.
간략히 설명하고 넘어갈 테니, MLE를 처음 들어보시는 분들은 ‘아 저런게 있구나’ 하고 넘어가 주셔도 무관합니다!<br/>

우리가 찾고자 하는 Parameter W와 b는, Unknown Parameter들입니다. 
분명 Optimal한 값들이 있는데, 지금은 그 값들이 무엇인지 알지 못하는 것이죠. 
이러한 Parameter들을 추정하는 것이 MLE의 목적이라고 볼 수 있습니다.<br/>

먼저, 계산된 확률을 토대로 Likelihood Function을 Construct할 수 있습니다. 
전체 Dataset에 대해, 각각의 Data의 정답이 나올 확률들을 모두 곱해주면 되겠죠? (Prediction들이 서로 독립이므로, 확률들을 단순히 곱해 주면 됩니다!) 
이제, 곱한 결과로 나온 Likelihood Function에 Log를 취해 줍니다. 그러면, Log의 성질에 의해, 확률들의 곱이 Log를 취한 확률들의 합으로 바뀌게 되겠죠?
이제, 이 Log Likelihood를 Maximize하는 Parameter W와 b를 얻고 싶습니다. 
Loss를 최소화하는 것이, Log Likelihood를 Maximize하는 것을 나타내면 되는거죠! 
그러기 위해서는, Log Likelihood에 (-1)을 곱해주면 됩니다. 
그러면, Loss를 최소화하는 것이 곧 Log Likelihood를 최대화하는 것이 되겠죠?<br/>

일반적으로, MLE를 구할 때는, Likelihood Function을 Parameter로 편미분하여 편미분 값이 0이 되는 지점의 Parameter를 찾습니다. 
하지만, 여기서 취한 Approach는 Likelihood Function의 최댓값을 찾을 때, 그것을 편미분하여 찾은 것이 아니라, Negative Log Likelihood가 최솟값을 갖는 지점을 Parameter의 값을 바꿔가며 찾는 것입니다!<br/>

Cross-Entropy Loss에서 나오는 MLE 개념은 더이상 자세하게 다루지는 않겠습니다. 
딥러닝 전체 개념의 흐름을 잡는데는 지장이 되지 않을 것 같습니다. 
저도 이 강의를 준비하면서 비로소 공부한 내용들이거든요. 
마찬가지로, 이해에 도움이 되었던 링크를 달아 놓겠습니다. 
[[Minimizing Negative Log Likelihood](http://jaejunyoo.blogspot.com/2018/02/minimizing-negative-log-likelihood-in-kor.html)] [[딥러닝 모델의 손실함수](https://ratsgo.github.io/deep%20learning/2017/09/24/loss/)]<br/>

지금까지 두 가지 Loss Function들을 배워 봤는데요, 어떤 Loss Function을 사용하는지도 학습 이전에 우리가 미리 결정해줘야 하는 사항입니다. 
따라서, 이것도 **Hyperparmeter**라 할 수 있겠죠? 
Multiclass SVM Loss가 잘 먹히는 상황이 있는 반면, Cross-Entropy Loss가 잘 먹히는 상황도 있습니다. 
뭐가 적합한지는 **Validation**을 통해서 찾아 주면 되는거죠!

<hr/>

## Regularization

이제 Loss 계산 식을 다시 볼까요?<br/>

우리의 목표는, Loss를 최소화하는 Parameter W와 b를 찾는 것입니다. 
Multiclass SVM Loss를 사용할 때, Loss = 0이 되게 하는 어떤 W’가 있다고 가정해 봅시다. 
그러면, 1 이상의 임의의 실수 k에 대해, k * W’도 Loss = 0을 만족하겠죠? 
점수를 k배만큼 Scale해주므로, 점수들간의 차도 k배만큼 Scale이 됩니다. 
W’에 대한 Loss가 0이었으므로, k * W’를 해주면, 정답 Label의 점수와 다른 Label의 점수 간의 간격이 더 커지니까 계속 Loss = 0이 되는 거죠!
따라서, Loss = 0인 W가 Unique하게 Determine되지 않습니다.<br/>

그렇다면, k * W’중에서 어떤 Parameter를 선택해야 좋은 Model을 만들 수 있을까요?<br/>

바로, 작은 값을 갖는 W를 선택하는 것입니다. 그 이유는 조금 후에 설명하겠습니다.<br/>

그렇다면, 작은 W를 더 선호하도록 Loss 계산식을 바꿔줘야 하겠죠? 
그래서 Loss Function 뒤에 **Regularization Term**이 붙습니다. 
Regularization Term은 W의 Entry들의 절댓값의 합, 혹은 제곱합 에 Regularization Strength를 곱한 것입니다. 
이 항을 붙임으로써, 큰 값들을 갖는 W의 Loss보다, 작은 값들을 갖는 W의 Loss가 더 작아집니다. 
따라서, 더 작은 W가 선호되는 것이죠.<br/>

그러면, 더 작은 W가 갖는 이점이 무엇일까요?<br/>

예시를 통해 살펴보겠습니다. 
Feature Vector X (1, 1, 1, 1)에 대해, 가중치 W1 (1, 0, 0, 0)과 가중치 W2 (.25, .25, .25, .25)가 있다고 해봅시다. 
X와 W1의 내적값과 X와 W2의 내적값은 1로 동일합니다. 
따라서, 1이라는 동일한 점수를 계산하게 되고, 다른 Label에 대해서도 마찬가지로 생각해 보면, 동일한 Prediction 결과와, 동일한 Loss Function 값을 가질 것입니다. 
W1과 W2의 차이는, 점수를 계산할 때, W1은 첫 번째 Feature에만 집중하여 점수를 계산하는 반면, W2는 가중치가 고르게 퍼져 있습니다. 
이 말은, W1에서는, 첫 번째 Feature가 점수에 대해 엄청나게 큰 영향을 미친다는 것입니다.<br/>

이렇게 한 Input Dimension이 점수 전체에 대해 큰 영향을 미치게 되면, 문제가 발생할 수 있습니다.
바로 Model이 Training Data의 Noise까지 불필요하게 학습할 수도 있다는 점입니다.
다른 말로 하면, Training Data에 과하게 적합한 Model을 만들 수도 있다는 것입니다. 
우리의 목표는, 실제 General한 Data에 대해서 예측을 잘 내리는, General한 Model을 만드는 것입니다. 
그렇기 때문에, 우리의 Model이 Training Data의 불필요한 Detail, 혹은 Noise까지 학습하여 Training Data에 맞춰진 Model이 되는 것을 피해야 합니다.<br/>

족보로 예시를 들 수 있을 것 같네요. 
동주는 지난 20년간 족보대로 시험문제를 출제하신 교수님 수업을 듣게 되었습니다. 
선배로부터 족보를 받은 동주는 다른 것은 공부하지 않고 오로지 족보만 공부했습니다. 
족보대로만 문제가 나온다면, 무조건 100점을 받을 수 있는 정도까지 공부를 했습니다. 
지금 동주는 족보라는 Training Data에 과하게 Fit된 Model이라고 할 수 있죠. 
하지만, 교수님이 족보가 돌아다닌다는 것을 인지하시고는, 새로운 유형의 문제들을 출제하셨습니다. 
이러한 General한 문제를 풀 수 없는 동주는 그만 시험에서 0점을 받고 말았습니다.<br/>

위의 예시처럼, Model이 Training Data에 과하게 맞춰지는 것을 **Overfitting** 현상이라고 합니다. 
반대로, Training이 충분히 이뤄지지 않은 Model은 **Underfit**되었다고 합니다. 
그래서 우리는 그 사이에서 Balance를 맞춰야 합니다. Underfit되지도 않고, Overfit되지도 않은 그 중간 지점에 Model이 가도록 해야 하죠!<br/>

항상 중간이 가장 어렵죠. 그래서 우리는 많은 시간을 균형 맞추기에 할애할 예정입니다.<br/>

그 균형 맞추기의 일환으로 Regularization Term이 붙은거죠! 
다시 W1, W2 예시로 돌아가면, L2 Regularizer는 더 고르게 Spread-Out된 Weight를 Favor함으로써, 특정 Input Dimension이 점수에 대해 너무 많은 영향을 미치는 것을 방지합니다. 
따라서, Overfitting을 방지하고 있다고 할 수 있는거죠.

<hr/>

## Review on Today’s Lecture

오늘은, 우리의 Model이 Train이 잘 되고 있는지, 현재 Parameter가 얼마나 적합한지를 수치적으로 나타내는 방법을 배웠습니다.<br/>

Loss가 Training Data에 대한 Parameter의 적합한 정도를 숫자로 나타낸 결과이고요,<br/>

Loss Function을 어떻게 정의하느냐에 따라 Loss 계산 방법이 달라지고, Optimal한 Model의 성질도 달라집니다.<br/>

또한, Overfitting을 방지하기 위해, Regularization Term을 붙여 주었습니다.<br/>

[[Linear Classifier](http://cs231n.github.io/linear-classify/)]를 읽어 보시면, 오늘 강의 내용 전반에 대해 더 잘 이해하실 수 있을거에요!

<hr/>

## Preview on Next Lecture

현재 Parameter가 얼마나 정확한지를 수치적으로 나타냈지만, 실제로 Parameter들을 Update하는 과정은 아직 다루지 않았습니다. 
그것은 다음 강의에서 다룰 예정입니다.<br/>

미리 맛보기를 하자면, Loss는 결국 Parameter W와 b에 대한 함수를 통과한 하나의 Scalar 결과값입니다. 
Loss의 최솟값으로 가기 위해서는, 더 낮은 지점으로 굴러 떨어질 필요가 있겠죠? 
이 굴러 떨어지는 방향을 결정하는 것이 바로 **Gradient**입니다. 
그리고, Loss라는 복잡한 함수의 Gradient를 계산하는 방법이 **Backpropagation**입니다.<br/>

오늘은 이 Preview를 정말 대강 이해하시면 됩니다. 다음 시간에 차근차근 다뤄 보도록 하겠습니다.
