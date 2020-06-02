---
title: Policy Gradient
date: 2020-06-03 03:00:00 +0000
categories: [스터디, PG study]
tags: [study, rl]     # TAG names should always be lowercase
---

## 시작하기 전에

[팡요랩 강화학습 7강](https://youtu.be/2YFBordM1fA) 의 내용을 보고 정리한것입니다.

### 3줄 요약

이건 졸려서 다음에 씀

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

* But 로컬 최저점에 빠지기 쉽고
* 비효율적이며 분산이 크다

### Policy Search

* 목적 함수 $J$ 3가지
  * expected return $E[R | \theta]$ 가 최대가 되도록
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
  고등학교떄 배운 $\log$미분 생각하면 된다.

<!-- Softmax Policy / Gaussian Policy 에서 φ(s)가 무엇? -->

### One-Step MDPs

한스탭 진행하고 종료되는 MDP

$$\begin{aligned}
  J(\theta) &= E_{\pi_\theta}[r] \\
  &= \sum_{s\in S}d(s)\sum_{a\in A}\pi_\theta(s,a)R_{s,a}
  \end{aligned} $$

* $E(x)=\sum_xxf(x)$
* 기대값 = 확률 * 확률변수 의 합
* $r$의 기대값 = $s$에서 $a$을 할 확률 * $r$ 의 합

$$\begin{aligned}
  \nabla_\theta J(\theta) &=
  \sum_{s\in S}d(s)\sum_{a\in A}\pi_\theta(s,a)\nabla_\theta\log\pi_\theta(s,a)R_{s,a} \\
  &= E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(s,a)r]
\end{aligned} $$

* 즉, $J$의 gradient를 기대값으로 표현 가능
* 지금 policy로 얻은 경험으로 $J$에 대한 gradient의 샘플을 얻는다

### Policy Gradient Theorem

* One-Step MDPs 를 multi-step MDPs로 확장한 개념
* 위의 식에서 $r$을 $Q$로 대체 한것

$$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(s,a)Q^{\pi_\theta}(s,a)]$$

* 증명은 생략

### REINFORCE

* 현실적으로 $Q$를 구할수 없음
* $Q^{\pi_\theta}(s,a)$의 unbiased sample인  retrun $G_t$를 사용
 
$$\Delta\theta_t=\alpha\nabla_\theta\log\pi_\theta(s,a)G_t$$

* But 분산이 여전히 너무 큼

> unbiased : 모평균이랑 샘플평균이랑 같은거

> 왜 unbiased **sample**인지?  
> 그렇다면 왜 **unbiased** 인지?  
> biased인지 아닌지는 어떻게 구하는지?
<!-- score function은 biased된것 방향성이 없어서? -->

## Actor-Critic Policy Gradient

* $Q_w(s,a)$를 $Q^{\pi_\theta}(s,a)$로 근사
* Policy iteration이랑 비슷한 방법
* $Q_w$ 업데이트는 지금까지 Value Function Approach 한거처럼 TD를 사용

$$\begin{aligned}
  \delta &= r+\gamma Q_w(s',a')-Q_w(s,a)\\
  \theta &= \theta+\alpha\nabla_\theta\log\pi_\theta(s,a)Q_w(s,a)
\end{aligned}$$

### Compatible Function Approximation

* policy gradient를 근사하는것은 bias하다
* 이떄 value function을 신중하게 근사하면 해결가능

> 솔직히 왜 policy가 bias한지 모르겠다  
> value function을 근사하면 해결가능하다는것도 이해안됨

이떄 
$$\nabla_w Q_w(s,a) = \nabla_\theta\log\pi_\theta(s,a)$$
$$\epsilon = E_{\pi_\theta}[(Q^{\pi_\theta}(s,a)-Q_w(s,a))^2]$$

하다고 할때,

$$\begin{aligned}
  \nabla_w\epsilon &= 0 \\
  E_{\pi_\theta}[(Q^{\pi_\theta}(s,a)-Q_w(s,a))\nabla_w Q_w(s,a)] &= 0 \\
  E_{\pi_\theta}[(Q^{\pi_\theta}(s,a)-Q_w(s,a))\nabla_\theta\log\pi_\theta(s,a)] &= 0 \\
  E_{\pi_\theta}[Q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(s,a)] &= E_{\pi_\theta}[Q_w(s,a)\nabla_\theta\log\pi_\theta(s,a)]
\end{aligned}$$

하므로

$$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(s,a)Q_w(s,a)]$$

$Q_w(s,a)$로 근사 시켜도 된다!

### Using Baseline

* $Q$에서 $V$를 뺀것
  
$$A^{\pi_\theta}(s,a)=Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s)$$

* 분산을 줄일수있다
* action사이의 상대적인 차이를 배울 수 있다
* 기대값은 안 바뀌면서 분산을 줄일수있다.

$$\begin{aligned}
  E_{\pi_\theta}[\nabla\log\pi_\theta(s,a)V^{\pi_\theta}(s)] &= \sum_{s\in S}d^{\pi_\theta}(s)\sum_{a\in A}\nabla_\theta\pi_\theta(s,a)V^{\pi_\theta}(s) \\
  &= \sum_{s\in S}d^{\pi_\theta}(s)V^{\pi_\theta}(s)\nabla_\theta\sum_{a\in A}\pi_\theta(s,a) \\
  &= 0
\end{aligned}$$

따라서 

$$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(s,a)A^{\pi_\theta}(s,a)]$$

> 마지막에 왜 0이 되는지 정확히 잘 모르겠다.  
> 1을 미분하면 0이라서?

### Estimating the Advantage Function

$Q$, $V$ 를 각각 따로 근사 시키지않고도 한번에 $A$를 구할 수 있다.

$$\delta^{\pi_\theta}=r+\gamma V^{\pi_\theta}(s')-V^{\pi_\theta}(s)$$

TD error $\delta$ 는 어드벤티지 함수의 unbiased 추정치 이므로

$$\begin{aligned}
  E_{\pi_\theta}[\delta^{\pi_\theta}|s,a] &= E_{\pi_\theta}[r+\gamma V^{\pi_\theta}(s')-V^{\pi_\theta}(s)|s,a]\\
  &= E_{\pi_\theta}[r+\gamma V^{\pi_\theta}(s')|s,a]-V^{\pi_\theta}(s)\\
  &= Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s)\\
  &= A^{\pi_\theta}(s,a)
\end{aligned}$$

따라서 TD error 를 써도 된다.

$$\nabla_\theta J(\theta)=E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(s,a)\delta^{\pi_\theta}(s,a)]$$

실제로는 $V_v$만 근사 시켜서 TD error를 계산한다.

$$\delta_v=r+\gamma V_v(s')-V_v(s)$$

> TD error 가 advantage의 unbiased estimate?  
> 델타 하나는 다르지만 평균을 취하면 A가 된다?

<!-- 분산, 샘플간의 독립성 -->

## 참고

[RLKorea sutton PG](https://reinforcement-learning-kr.github.io/2018/06/28/1_sutton-pg/)  
[기댓값](https://blog.naver.com/mykepzzang/220837877074)