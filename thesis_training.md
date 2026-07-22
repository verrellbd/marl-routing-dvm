## Training

Training optimises the shared policy parameters $\theta$ and the centralised critic parameters
$\phi$ against a fast analytical model of the network, and never invokes the packet-level
simulator inside the learning loop. The surrogate is exactly the load model of the previous
section: given the paths the agents have committed so far, arc loads and utilisations are
$\boldsymbol{\ell} = \mathbf{R}^{\!\top}\mathbf{r}$ and $u_e = 100\,\ell_e/c_e$, from which the
bottleneck $U_{\max}$ and hence the reward follow in closed form. This makes an episode cost a
few milliseconds instead of the minutes an ns-3 run requires, which is what makes on-policy
learning tractable; ns-3 is reserved for judging the finished policy, so the simulator is never
part of the gradient path.

An episode is a complete routing of one demand matrix: the flows with $r_i > 0$ are ordered by
decreasing rate and routed hop by hop, so an episode consists of
$K = \sum_i L_i$ decisions and terminates when the last flow reaches its destination. Because the
per-hop reward telescopes, the undiscounted episode return is exactly the negated objective,

$$
\sum_{k=0}^{K-1} R^{(k)} \;=\; -\,U_{\max} \;-\; \beta \sum_i \bigl(L_i - L_i^{\mathrm{sp}}\bigr),
$$

so no separate terminal bonus is needed and the quantity being maximised during training is the
same quantity on which OSPF and the greedy best-response reference are scored.

The training objective is the expected return over a distribution $\mathcal{D}_{\mathrm{train}}$
of demand matrices rather than a single traffic snapshot,

$$
J(\theta) \;=\; \mathbb{E}_{\mathbf{r} \sim \mathcal{D}_{\mathrm{train}}}\;
\mathbb{E}_{\tau \sim \pi_\theta}\Bigl[\textstyle\sum_{k} \gamma^{k} R^{(k)}\Bigr],
\qquad
\mathcal{D}_{\mathrm{train}} = \bigl\{\, \alpha \cdot \mathbf{r}^{(t)} \;:\;
\alpha \in \mathcal{A},\; t \in \mathcal{T}_{\mathrm{train}} \,\bigr\},
$$

where $\mathcal{T}_{\mathrm{train}}$ indexes the earlier portion of the measured SNDlib demand
series and $\mathcal{A}$ is a set of magnitude scales that place the network in the congested
regime in which routing decisions matter ($\mathcal{A} = \{2,3,4\}$ by default; at low load every
routing is feasible and no policy can separate itself from OSPF). Each episode draws one matrix
uniformly from $\mathcal{D}_{\mathrm{train}}$, so the policy is trained across both traffic
patterns and load levels rather than being fitted to one operating point. Evaluation uses matrices
from the disjoint later window $\mathcal{T}_{\mathrm{test}}$, so the reported numbers measure
temporal generalisation.

Learning proceeds by repeated cycles of on-policy collection and clipped policy improvement.
Each iteration collects a fixed budget of $T = 4096$ decision steps under
$\pi_{\theta_{\mathrm{old}}}$, resetting to a freshly drawn matrix whenever an episode ends, and
records for every step the local observation $o^{(k)}$, the action mask $\mathbf{m}^{(k)}$, the
global critic state $s^{(k)}$, the action, its log-probability and the critic value. Advantages
are computed by generalised advantage estimation, bootstrapping the value of the state at which
collection was truncated,

$$
\delta^{(k)} = R^{(k)} + \gamma\,V_\phi\bigl(s^{(k+1)}\bigr)\bigl(1 - \mathrm{done}^{(k)}\bigr) - V_\phi\bigl(s^{(k)}\bigr),
\qquad
\hat{A}^{(k)} = \sum_{j=0}^{T-1-k} (\gamma\lambda)^{j}\,\delta^{(k+j)},
$$

with value targets $\hat{V}^{(k)} = \hat{A}^{(k)} + V_\phi(s^{(k)})$. Advantages are standardised
across the batch before the update,

$$
\tilde{A}^{(k)} = \frac{\hat{A}^{(k)} - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}},
$$

which keeps the gradient scale independent of the load level of the matrices that happened to be
sampled — necessary here because episode returns differ by an order of magnitude between
$\alpha = 2$ and $\alpha = 4$.

The collected batch is then reused for $E$ epochs of minibatch stochastic gradient descent on the
combined actor–critic loss

$$
\mathcal{L}(\theta,\phi) \;=\;
\underbrace{-\,\mathbb{E}_{k}\Bigl[\min\bigl(\rho^{(k)}\tilde{A}^{(k)},\;
\mathrm{clip}(\rho^{(k)}, 1-\varepsilon, 1+\varepsilon)\,\tilde{A}^{(k)}\bigr)\Bigr]}_{\text{policy}}
\;+\; c_v \underbrace{\mathbb{E}_{k}\Bigl[\bigl(V_\phi(s^{(k)}) - \hat{V}^{(k)}\bigr)^{2}\Bigr]}_{\text{value}}
\;-\; c_e \underbrace{\mathbb{E}_{k}\Bigl[\mathcal{H}\bigl[\pi_\theta(\cdot \mid o^{(k)}, \mathbf{m}^{(k)})\bigr]\Bigr]}_{\text{entropy}},
$$

with importance ratio

$$
\rho^{(k)} \;=\; \frac{\pi_\theta\bigl(a^{(k)} \mid o^{(k)}, \mathbf{m}^{(k)}\bigr)}
{\pi_{\theta_{\mathrm{old}}}\bigl(a^{(k)} \mid o^{(k)}, \mathbf{m}^{(k)}\bigr)} .
$$

The action mask enters both the behaviour and the target policy, so $\rho^{(k)}$ is evaluated over
the same admissible next-hop set that produced the action and the clipped surrogate remains a
valid off-policy correction. The entropy term is computed over the masked distribution only, so it
encourages exploration among *legal* next hops and cannot push probability mass onto hops that the
loop-freedom constraints forbid. Parameters are updated with Adam under global gradient-norm
clipping,

$$
\mathbf{g} \leftarrow \nabla_{\theta,\phi}\mathcal{L},
\qquad
\mathbf{g} \leftarrow \mathbf{g}\cdot\min\!\left(1, \frac{g_{\max}}{\lVert \mathbf{g} \rVert_2}\right),
\qquad
(\theta,\phi) \leftarrow \mathrm{Adam}\bigl((\theta,\phi),\,\mathbf{g},\,\alpha_{\mathrm{lr}}\bigr),
$$

which is required because the reward is a max over arcs and a single hop that creates a new
bottleneck produces a large, sparse gradient. Since the actor and critic are separate networks
sharing no trunk, the coefficient $c_v$ only balances the two gradient magnitudes and does not
trade capacity between the heads. The whole cycle repeats for
$\lfloor N_{\mathrm{steps}}/T \rfloor$ iterations.

Parameter sharing is what makes this a single optimisation problem rather than $N$ of them: all
router agents update the same $\theta$, and every decision taken anywhere in the network
contributes a gradient term, so the effective sample size per iteration is $T$ decisions rather
than $T/N$ per-agent trajectories. Agent identity is supplied through the one-hot fields of
$o_v$, so the shared policy can still specialise its behaviour by location while generalising
across routers with similar local structure.

| Symbol | Meaning | Value |
|---|---|---|
| $N_{\mathrm{steps}}$ | total decision steps | $6\times10^{5}$ |
| $T$ | rollout length per iteration | $4096$ |
| $E$ | epochs per batch | $8$ |
| $B$ | minibatch size | $512$ |
| $\alpha_{\mathrm{lr}}$ | Adam learning rate | $3\times10^{-4}$ |
| $\gamma$ | discount | $0.99$ |
| $\lambda$ | GAE trace | $0.95$ |
| $\varepsilon$ | PPO clip range | $0.2$ |
| $c_v$ | value coefficient | $0.5$ |
| $c_e$ | entropy coefficient | $0.01$ |
| $g_{\max}$ | gradient-norm clip | $0.5$ |
| $H$ | hidden width | $128$ |
| $\beta$ | detour penalty | $0.5$ |
| $\sigma$ | stretch | $1$ |

Progress is monitored on a held-out set $\mathcal{H}$ of unseen demand matrices, using the
deterministic policy $a = \arg\max_a \pi_\theta(a \mid o, \mathbf{m})$ so that the metric reflects
what would be deployed rather than a sampled trajectory. The two quantities tracked are the mean
bottleneck improvement over OSPF and the fraction of matrices on which the policy strictly wins,

$$
\overline{\Delta} = \frac{1}{|\mathcal{H}|}\sum_{\mathbf{r} \in \mathcal{H}}
\bigl(U^{\mathrm{OSPF}}_{\max}(\mathbf{r}) - U^{\pi_\theta}_{\max}(\mathbf{r})\bigr),
\qquad
w = \frac{1}{|\mathcal{H}|}\Bigl|\bigl\{\mathbf{r} \in \mathcal{H} :
U^{\pi_\theta}_{\max}(\mathbf{r}) < U^{\mathrm{OSPF}}_{\max}(\mathbf{r}) - \varepsilon_0 \bigr\}\Bigr|,
$$

with a tolerance $\varepsilon_0 = 0.5$ percentage points so that numerically insignificant
differences are counted as ties rather than wins. The greedy myopic best-response — at each hop
take the admissible neighbour minimising the resulting bottleneck — is evaluated on the same
matrices as an intermediate reference between OSPF and the learned policy: it bounds what purely
local, non-anticipatory decision-making achieves, so the gap between greedy and $\pi_\theta$
isolates the value of the learned, congestion-anticipating behaviour. Checkpoints are written
periodically and the final parameters are exported for ns-3 evaluation, where the same
deterministic rollout produces the explicit per-demand paths that are installed as static routes.
Each configuration is trained from several independent seeds, and all reported results are given
with across-seed variance.
