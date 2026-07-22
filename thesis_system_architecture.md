## System architecture

The system is designed as a pipeline of independent components that share a unified source of
truth for both network topology and traffic. A network is treated as two environments that vary
independently in operational practice: the *topology*, i.e. the physical point-to-point
connectivity between routers together with the capacity of each link, and the *traffic*, i.e.
the time-varying demand offered between every pair of routers. Both are drawn from the SNDlib
repository, which publishes real backbone topologies alongside measured demand-matrix time
series; Abilene, GÉANT and Germany50 are used as the foundational environments. Each instance is
parsed once into a canonical JSON description that is consumed unchanged by the topology loader,
the traffic loader, the learning environment, the OSPF baseline and the ns-3 simulator, so the
agent, the baseline and the evaluator are guaranteed to reason about exactly the same network.
Policies are trained against a fast analytical link-utilisation model of that network, and every
reported result is produced by installing the resulting paths as static routes in ns-3 and
measuring them at the packet level.

Formally, the router-level network is a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$
with $|\mathcal{V}| = N$ routers and $|\mathcal{E}| = M$ arcs, each undirected physical link
being instantiated as two arcs so that the two directions carry load independently. Every arc
$e$ has a capacity $c_e > 0$ taken from the SNDlib instance. Connectivity enters the model
through the adjacency matrix $\mathbf{A} \in \{0,1\}^{N \times N}$, with $A_{uv} = 1$ iff
$(u,v) \in \mathcal{E}$, and the node–arc incidence matrix $\mathbf{B} \in \{0,1\}^{N \times M}$,
with $B_{ve} = 1$ iff router $v$ is the tail or the head of arc $e$. Traffic at a given instant
is a demand matrix $\mathbf{D} \in \mathbb{R}^{N \times N}_{\ge 0}$, whose entry $D_{sd}$ is the
rate offered from $s$ to $d$; flattening the ordered pair set $\mathcal{P} = \{(s,d): s \ne d\}$
gives the demand vector $\mathbf{r}$ with $r_i = D_{s_i d_i}$. The SNDlib series supplies a
sequence $\{\mathbf{D}^{(t)}\}_{t=1}^{T}$ of such matrices, of which earlier matrices are used
for training and later, unseen matrices for evaluation. A routing assigns to each demand $i$ a
single loop-free path $\pi_i = (s_i = v_0, v_1, \dots, v_{L_i} = d_i)$ with
$(v_j, v_{j+1}) \in \mathcal{E}$, matching the unsplittable static-route model installed in ns-3.
Writing $\mathbf{R} \in \{0,1\}^{|\mathcal{P}| \times M}$ for the induced path–arc indicator
matrix, the arc loads, utilisations and network bottleneck are

$$
\boldsymbol{\ell} = \mathbf{R}^{\!\top}\mathbf{r},
\qquad
u_e = 100 \cdot \frac{\ell_e}{c_e}\,[\%],
\qquad
U_{\max} = \max_{e \in \mathcal{E}} u_e ,
$$

and the control objective is to minimise the bottleneck under a bounded path-length penalty,

$$
\min_{\{\pi_i\}} \; U_{\max} \;+\; \beta \sum_{i} \bigl(L_i - L_i^{\mathrm{sp}}\bigr),
$$

where $L_i^{\mathrm{sp}}$ is the shortest-path hop count and $\beta \ge 0$ prices each
unnecessary detour, so that the optimum collapses onto shortest-path routing whenever the
network is uncongested. OSPF is the special case in which every $\pi_i$ is the shortest path and
$\mathbf{R}$ is therefore fixed independently of $\mathbf{r}$.

Rather than solving this assignment in one shot, the architecture expresses routing as a
sequential decision process over hops: demands are processed in decreasing order of rate, and at
the router the traffic has currently reached the resident agent selects the next hop, whose load
is committed immediately, so that later decisions observe the congestion earlier ones created.
Loop-freedom is guaranteed structurally rather than learned. With $h_d(v)$ the shortest-path hop
distance from $v$ to $d$, $\mathcal{S}$ the set of routers already visited by the current demand
and $k$ the hops taken so far, the admissible next hops at $(v, d)$ are

$$
\mathcal{N}^{\mathrm{ok}}(v,d) = \bigl\{\, m \in \mathcal{N}(v) \;:\; m \notin \mathcal{S},\;\;
h_d(m) < h_d(v) + \sigma,\;\; k + 1 + h_d(m) \le L^{\mathrm{sp}} + \sigma_{\max} \,\bigr\},
$$

where the stretch $\sigma$ controls how far a hop may move away from the destination
($\sigma = 1$ confines the agent to the destination-oriented shortest-path DAG) and
$\sigma_{\max}$ caps the excess length of the final path; the visited set and the finite router
count together make every trace simple and terminating. Committing a hop on arc $e$ yields the
reward

$$
R^{(k)} = -\bigl(U^{(k+1)}_{\max} - U^{(k)}_{\max}\bigr) - \beta\,\mathbb{1}\!\left[h_d(m) \ge h_d(v)\right],
$$

whose return telescopes over a full episode to $-U_{\max} - \beta\,(\text{detour hops})$, so that
maximising return is exactly minimising the objective above and the learned policy, the OSPF
baseline and a greedy best-response reference are all scored on the same quantity.

The policy backbone is a graph neural network defined directly on the router-level topology, so
that a single parameter set describes a network of any size and the learned function respects its
connectivity. The per-arc utilisation vector $\mathbf{u} \in \mathbb{R}^{M}$ is first lifted onto
the routers through the incidence structure and then mixed with each router's immediate
neighbourhood through the degree-normalised adjacency,

$$
\mathbf{x}^{(0)} = \tilde{\mathbf{B}}\mathbf{u},
\quad \tilde{B}_{ve} = \frac{B_{ve}}{\max\bigl(1, \textstyle\sum_{e'} B_{ve'}\bigr)};
\qquad
\mathbf{x}^{(1)} = \hat{\mathbf{A}}\mathbf{x}^{(0)},
\quad \hat{\mathbf{A}} = \mathbf{\Lambda}^{-1}\mathbf{A},
\quad \Lambda_{vv} = \textstyle\sum_{w} A_{vw} + \epsilon,
$$

so that each router is summarised by the mean utilisation of its incident arcs and then by that
of its neighbours. The mixed states are concatenated with the vectorised adjacency — conditioning
the encoder on the topology it is operating over — and passed through fully connected layers to
form the graph embedding $\mathbf{z} = \phi\bigl([\mathbf{x}^{(1)} \Vert \mathrm{vec}(\mathbf{A})]\bigr)
\in \mathbb{R}^{H}$, which is combined with the decision-local context $\mathbf{q}$ (one-hot
encodings of the current router and destination, the normalised demand rate, and per-neighbour
features comprising arc utilisation, the bottleneck that would result from using the arc,
remaining headroom and normalised distance to the destination) into the shared representation
$\mathbf{f} = \psi\bigl([\mathbf{z} \Vert \mathbf{q}]\bigr)$. An actor head and a critic head read
$\mathbf{f}$; the actor emits next-hop logits $\boldsymbol{\eta} = \mathbf{W}_\pi \mathbf{f} +
\mathbf{b}_\pi$ from which the inadmissible hops are removed by the binary mask
$\mathbf{m} \in \{0,1\}^{\Delta}$ induced by $\mathcal{N}^{\mathrm{ok}}(v,d)$, with $\Delta$ the
maximum router degree,

$$
\pi_\theta(a \mid o) = \frac{\exp(\eta_a + \log m_a)}{\sum_{a'} \exp(\eta_{a'} + \log m_{a'})},
$$

so that $\pi_\theta(a \mid o) = 0$ whenever $m_a = 0$ and every sampled action is admissible by
construction. Because $\tilde{\mathbf{B}}$ and $\hat{\mathbf{A}}$ are the only topology-dependent
objects and both are supplied as inputs rather than learned, the same parameters $\theta$ are
shared by every router agent and, in the topology-agnostic variant, across topologies of
different sizes.

The resulting problem is a decentralised partially observable Markov decision process in which
each router is an agent: agent $v$ observes only the local vector $o_v$ described above and
chooses a next hop $a_v \in \mathcal{N}^{\mathrm{ok}}(v,d)$, never seeing the global link state at
execution time, which is what makes the controller deployable as a distributed control plane.
The policy is trained with a custom multi-agent proximal policy optimisation (MAPPO) algorithm
implementing centralised training with decentralised execution: during training a centralised
critic is given the global state
$s = [\mathbf{u} \Vert \mathbf{e}_v \Vert \mathbf{e}_d \Vert r/r_0 \Vert U_{\max}]$, containing
the utilisation of every arc in the network, which resolves the credit-assignment problem created
by agents whose local views overlap only partially. Advantages are computed by generalised
advantage estimation over the sequence of hop decisions,
$\delta^{(k)} = R^{(k)} + \gamma V_\phi(s^{(k+1)})(1 - \mathrm{done}^{(k)}) - V_\phi(s^{(k)})$ and
$\hat{A}^{(k)} = \sum_{j \ge 0} (\gamma\lambda)^{j}\delta^{(k+j)}$, and the shared parameters are
updated with the clipped surrogate objective

$$
\mathcal{L}(\theta,\phi) = \mathbb{E}\Bigl[\min\bigl(\rho^{(k)}\hat{A}^{(k)},\,
\mathrm{clip}(\rho^{(k)}, 1-\varepsilon, 1+\varepsilon)\hat{A}^{(k)}\bigr)
- c_v\bigl(V_\phi(s^{(k)}) - \hat{V}^{(k)}\bigr)^{2}
+ c_e\,\mathcal{H}\bigl[\pi_\theta(\cdot \mid o^{(k)})\bigr]\Bigr],
$$

with $\rho^{(k)} = \pi_\theta(a^{(k)}\mid o^{(k)})/\pi_{\theta_{\mathrm{old}}}(a^{(k)}\mid o^{(k)})$
and $\hat{V}^{(k)} = \hat{A}^{(k)} + V_\phi(s^{(k)})$. The critic is discarded after training,
leaving a policy that each router evaluates on its own observation; at evaluation it is rolled
out deterministically to produce an explicit path per demand, which — together with the OSPF
shortest paths for the same demand matrix — is installed in ns-3 and measured under identical
packet-level conditions, so that the comparison isolates the routing decision itself.
