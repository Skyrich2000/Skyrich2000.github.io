---
title: Policy Gradient
date: 2020-06-03 03:00:00 +0000
categories: [스터디, PG study]
tags: [study, rl]     # TAG names should always be lowercase
---

## 시작하기 전에

[팡요랩 강화학습 7강](https://youtu.be/2YFBordM1fA) 의 내용을 보고 정리한것입니다.

## Introduction of Policy Gradient

### Value Function Approach

* 지금까지 강화학습은 value function을 기반으로 동작
* 이런 방법은 deterministic policy를 구함 (e.g. $\epsilon$-greedy)
* stochastic policy, high-dimensional, continuous action spaces 를 위해
* policy search라는 새로운 방법이 나왔다.

> deterministic : 특정 state에 대해 결정될 action이 정해져있음  
> stochastic : 특정 state에 대해 결정될 action이 확률적으로 정해짐
>  
> ex) 가위바위보, Aliased Gridworld

* But
* 로컬 최저점에 빠지기 쉽다
* 비효율적이고 분산이 크다

### Policy Search

* 목적 함수 $J$ 3가지
  * expected return $E[R | \theta]$가 최대가 되도록
  * start value
    $$J(\theta)=V^{\pi_\theta}(S_1)=E_{\pi_\theta}[v_1]$$
  * average value
    $$J(\theta)=\sum_sd^{\pi_\theta}(s)V^{\pi_\theta}(s)$$
  * average reward per time-step
    $$J(\theta)=\sum_sd^{\pi_\theta}(s)\sum_a\pi_\theta(s,a)R^a_s$$

  > stationary distribution $d^{\pi_\theta}(s)$

* 최적화
  * $\theta$를 조절해서 $J$를 최적화
  * Gradient Ascent 사용
  * $J$의 gradient를 구해서 $\theta$를 조절하여 $\pi^*$를 구함

* Obtain the Expected Return
  * deterministic approximation : 다이나믹 프로그래밍으로 수식을 통해 구함
  * monte carlo estimation : 많은 sample로 경험에 의해 expected return을 계산

## Monte-Carlo Policy Gradient

* 가정
  * $\pi_\theta$ 가 미분 가능하다
  * $\nabla_\theta\pi_\theta(s,a)$ 를 안다.
  
* Likelihood ratios trick
  $$\begin{aligned}
    \nabla_\theta\pi_\theta(s,a) &= \pi_\theta(s,a)\cfrac{\nabla_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)} \\
    &= \pi_\theta(s,a)\nabla_\theta\log\pi_\theta(s,a)
  \end{aligned}$$

<!-- Softmax Policy / Gaussian Policy 에서 φ(s)가 무엇? -->

### One-Step MDPs

* 한스탭 진행하고 종료되는 MDP

![](/assets/img/2020-05-17-policy-gradient/2020-05-17-policy-gradient_203636.png)

* 시그마 두개가 기댓값의 다른 표현
* 왜? 기댓값은 각 확률 곱하기 변수 의 합
* J의 대한 gradient가 기대값으로 표현 가능
* 즉, 지금 policy를 따라 가면서 얻은 경험으로 gradient를 구할수있다.
* J에 대한 gradient의 샘플을 얻는것

### Policy Gradient Theorem

* One-Step MDPs 를 multi-step MDPs로 확장한 개념
* 위의 식에서 $r$을 $Q$로 대체 한것
* $\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(s,a)Q^{\pi_\theta}(s,a)]$
* 증명은 생략

### REINFORCE

> unbiased : 모평균이랑 샘플평균이랑 같은거
<!-- 이게 맞음?? unbiased 인지 아닌지는 어떻게 구함? -->
<!-- score function은 biased된것 방향성이 없어서? -->

* $Q$를 구할수 없음
* $Q^{\pi_\theta}(s,a)$의 unbiased sample인  retrun $G_t$를 사용
* But 분산이 너무 큼

## Actor-Critic Policy Gradient

* 실제 $Q$를 학습 시키자
* Policy iteration이랑 비슷한 방법
* $Q$ 업데이트는 지금까지 Value Function Approach 한거처럼 TD를 사용

<!-- Q_w 로 approximate 해도 된다 -> 증명 확인 -->

### Using Baseline

* 분산을 줄이고 action사이의 상대적인 차이를 배우게 하기 위해
* 기대값은 안 바뀌면서 분산을 줄일수있다.
<!-- V를 빼도 수학적으로 맞다느걸 증명하는거에서 왜 0이 되는지? 1을 미분해서 0이 된다?-->

### Estimating the Advantage Function

* TD error 가 advantage의 unbiased sample이다? 왜 sample?

* 델타 하나는 다르지만 평균을 취하면 A가 된다 ?
  
<!-- Q에만 씌운ㅇ 이유 - 뒤에꺼는쓰우나 마나 똑같아서  -->
<!-- 분산, 샘플간의 독립성 -->

## 참고

[RLKorea sutton PG](https://reinforcement-learning-kr.github.io/2018/06/28/1_sutton-pg/)