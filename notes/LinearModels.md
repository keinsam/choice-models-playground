# Notes : Linear Models

## **Random Utility Models (RUM)**

**Objective:**

We aim to model the probability that an agent $i$ chooses alternative $d$ among a discrete set of $J$ options.
This is done by specifying a utility model and computing the probability that alternative $d$ yields the highest utility for agent $i$.

**Definition (Random Utility Model or RUM):**

For every agent $i$, we define the utility vector
$$
\mathbf{U}_i = (U_{i1}, \dots, U_{iJ}) = \mathbf{v}_i + \boldsymbol{\varepsilon}_i,
$$
where:

- $\mathbf{v}_i = (v_{i1}, \dots, v_{iJ}) \in \mathbb{R}^J$ is the deterministic utility vector for each of the $J$ alternatives,
- $\boldsymbol{\varepsilon}_i$ is a random noise vector in $\mathbb{R}^J$ (distribution not specified at this stage),
- $\mathbf{U}_i \in \mathbb{R}^J$ is the total utility vector.

Then, the agent chooses the alternative with the highest utility:

$$
y_i = \arg\max_{j \in \{1, \dots, J\}} U_{ij}
$$

**Property (Conditional Choice Probability or CCP):**

Let $\boldsymbol{\varepsilon}_i \sim \text{Gumbel}(\mathbf{0}_J, \mathbf{1}_J)$ with i.i.d. components. Then,

$$
\Pr(y_i = d \mid \mathbf{v}_i) = \frac{\exp(v_{id})}{\sum_{j=1}^{J} \exp(v_{ij})}
$$

**Property (Independance of Irrelevant Alternatives or IIA):**

Let $\boldsymbol{\varepsilon}_i \sim \text{Gumbel}(\mathbf{0}_J, \mathbf{1}_J)$ with i.i.d. components. Then,

$$
\frac{\Pr(y_i = j \mid \mathbf{v}_i)}{\Pr(y_i = k \mid \mathbf{v}_i)} = \exp(v_{ij} - v_{ik})
$$

## **Conditional Logit**

**Definition (Conditional Logit or CL):**

Conditional Logit is a **RUM** with $\mathbf{v}_i = X \boldsymbol{\beta}$, where:

- $X \in \mathbb{R}^{J \times K}$ is the matrix of alternative features (same for all individuals),
- $\boldsymbol{\beta} \in \mathbb{R}^{K}$ is the common coefficient vector applied to all alternatives,
- $\mathbf{v}_i \in \mathbb{R}^{J}$ is the deterministic utility vector for individual $i$,
- $\boldsymbol{\varepsilon}_i \sim \text{Gumbel}(\mathbf{0}_J, \mathbf{1}_J)$ is i.i.d. across alternatives,
- $J$ is the number of alternatives, $K$ is the number of features per alternative.

*Under the i.i.d. Gumbel noise assumption, the conditional logit model satisfies both the **CCP** and **IIA** properties.*

## **Multinomial Logit**

**Definition (Multinomial Logit or MNL):**

Multinomial Logit is a **RUM** with $\mathbf{v}_i = B \mathbf{x}_i$, where:

- $\mathbf{x}_i \in \mathbb{R}^{K}$ is the feature vector of individual $i$ (same across alternatives),
- $B \in \mathbb{R}^{J \times K}$ is the matrix of coefficients (one row per alternative),
- $\mathbf{v}_i \in \mathbb{R}^{J}$ is the deterministic utility vector for individual $i$,
- $\boldsymbol{\varepsilon}_i \sim \text{Gumbel}(\mathbf{0}_J, \mathbf{1}_J)$ is i.i.d. across alternatives,
- $J$ is the number of alternatives, $K$ is the number of features per alternative.

*Under the i.i.d. Gumbel noise assumption, the multinomial logit model satisfies both the **CCP** and **IIA** properties.*

## **Nested Logit**