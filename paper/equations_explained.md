# How the equations work, step by step

A plain-language walkthrough of the Methodology chapter, from the network model to
MARL with a GNN backbone. Each step gives the equation as it appears in the thesis,
the same thing in ordinary words, and why it is there.

**The metaphor used throughout:** a city of road junctions, with a traffic officer
standing at each one. Parcels (flows) need to get across the city, and the officers
decide which road each parcel takes.

---

## Step 1 — Measure how bad things are

**Thesis: Eq. (1) `eq:load` and Eq. (2) `eq:bottleneck`**

$$\ell_e=\sum_{f\in\mathcal{F}} r_f\,\mathbb{1}[e\in p_f], \qquad u_e=100\,\frac{\ell_e}{c_e}\quad[\%]$$

$$U=\max_{e\in\mathcal{E}} u_e$$

In words:

```
load on road e   = sum of the rates of all parcels using road e
fullness of e    = 100 × (load on e) ÷ (width of e)
U                = the largest fullness among all roads
```

**U is the fullness of the single worst road in the city.** Not the average — the worst.

A city is only as good as its worst jam. If one road sits at 150% and everything else
is empty, packets die on that road while the average looks fine. U is the number the
whole project tries to make small.

---

## Step 2 — Write the rules of the game

**Thesis: Eq. (3) `eq:mask`**

$$\mathcal{A}^{\mathrm{ok}}_i(d)=\bigl\{\,j\in\mathcal{A}_i \;:\;
j\notin\mathcal{V}_f,\;\; h_d(j)<h_d(i)+\sigma,\;\;
\omega_t+w_{ij}+h_d(j)\le h^{\star}_f+\sigma_{\max}\bigr\}$$

In words:

```
a road to neighbour j is legal only if:
  1. the parcel has never visited j
  2. j is not much further from the destination than where I am now   (allowance σ)
  3. cost so far + this road + rest of the way ≤ shortest route + allowance σ_max
```

Rule 1 makes loops **impossible**, not merely discouraged — the parcel can never
revisit a junction. Rules 2 and 3 stop it wandering.

This is a fence around the officer's choices, enforced before it picks rather than
learned. Many RL routing papers *hope* the agent learns to avoid loops. This one
physically cannot make one.

---

## Step 3 — Decide how to score the officers

This is the heart of the method.

**Thesis: Eq. (4) `eq:reward`** — the score for one hop:

$$r_t=-\,\frac{100\,(U_t-U_{t-1})}{U^{\mathrm{OSPF}}}\;-\;\beta\,\mathbb{1}[\text{hop } t \text{ is a detour}]$$

In words:

```
score for this hop = −100 × (U_now − U_before) ÷ U_ospf    ← congestion charge
                     − 0.5  if this hop was a detour        ← detour fine
```

Two charges. The congestion charge bills you for **how much worse you made the worst
road** — the change you caused, not the total. The detour fine is a flat β = 0.5
whenever a hop fails to move the parcel closer to its destination.

**Thesis: Eq. (5) `eq:telescope`** — add up every hop and the intermediate values cancel:

$$\sum_{t} r_t=-\,\frac{100\,U_T}{U^{\mathrm{OSPF}}}\;-\;\beta\sum_{t}\mathbb{1}[\text{hop } t \text{ is a detour}]$$

In words:

```
total for the day = −100 × U_final ÷ U_ospf  −  0.5 × (total number of detour hops)
```

> **Metaphor.** Bill each officer only for the *extra* water they add to a bucket. At
> the end of the day those small bills sum to exactly the total water in the bucket.
> Nobody had to measure the bucket — the running total already got it right.

Two things come free from this:

- **Feedback arrives every hop** instead of only at the end of the episode. Waiting
  until the end is how reinforcement learning usually fails on long tasks.
- **The score reads directly.** −100 means "exactly as good as OSPF, with no
  detours". So −75 is better than OSPF and −108 is worse. This is why the training
  curve is interpretable at a glance.

Dividing by U_ospf matters too: it puts every city on the same scale, so a small
town and a huge metropolis both score around −100 for OSPF-equivalent work. That is
what lets **one policy learn from all seventeen training cities at once** without the
big ones drowning out the small ones.

---

## Step 4 — Decide what each officer may see

**Thesis: Eq. (6) `eq:nodefeat`**

$$o_i(t)=\bigl[\,\mathbb{1}[i{=}c_t],\;\mathbb{1}[i{=}d_f],\;
\tfrac{h_{d_f}(i)}{h_{\max}},\;
\tfrac{1}{100}\max_{e\in\mathrm{out}(i)} u_e,\;
\tfrac{1}{100}\min_{e\in\mathrm{out}(i)}(100-u_e),\;
\tfrac{\deg(i)}{\Delta}\,\bigr]\in\mathbb{R}^{6}$$

In words — each junction is exactly six numbers:

```
[ am I holding the parcel?      (1 or 0)
  am I the destination?         (1 or 0)
  how far am I from it          (0 to 1)
  my fullest road               (0 to 1)
  my emptiest road's headroom   (0 to 1)
  how many roads I have         (0 to 1) ]
```

**Notice what is missing: the junction's name, and where it sits on the map.**

This is the most important design choice in the project. The officer never knows "I
am junction 7 in Chicago". It knows only "I am holding a parcel, I am 3 hops out, my
busiest road is 80% full".

If it knew names, a policy trained in one city would be useless in another — junction
7 would not exist there. Because it knows only *role and pressure*, **the same officer
can be dropped into a city it has never seen and still work.** The zero-shot result
comes from here.

---

## Step 5 — Let officers see past their own junction

**Thesis: Eq. (7) `eq:gnn`**

$$h^{(0)}_i=\mathrm{ReLU}(W_{\mathrm{enc}}\,o_i(t)),
\qquad
h^{(l)}_i=h^{(l-1)}_i+\mathrm{ReLU}\bigl(W^{(l)}[\,h^{(l-1)}_i \,\Vert\, \bar{A}_i h^{(l-1)}\,]\bigr)$$

In words — the officers gossip, three rounds:

```
round 1:  new me = old me + f(old me, average of my neighbours)
round 2:  same again
round 3:  same again
```

An officer seeing only their own roads is short-sighted: they might pick a clear road
that leads straight into a jam two junctions away. After three rounds every officer
senses conditions **three junctions out in all directions**, with nobody needing a
view of the whole city.

The "old me +" part is a residual connection. It means *keep what you already knew,
and add what you just heard* — it stops officers forgetting their own situation while
listening to gossip.

---

## Step 6 — Make the choice

**Thesis: Eq. (8) `eq:policy`**

$$\eta_j=\mathrm{MLP}_\theta[\,h^{(L)}_{c_t}\Vert h^{(L)}_{j}\Vert \ell_{c_tj}\,],
\qquad
\pi_\theta(j\mid o,m)=\frac{m_j\exp\eta_j}{\sum_{j'}m_{j'}\exp\eta_{j'}}$$

In words:

```
score for road j = small_network( my state , neighbour j's state , this road's 4 numbers )

chance of choosing j = exp(score j) ÷ sum of exp(scores of all LEGAL roads)
```

The mask from Step 2 multiplies illegal roads to zero, so they can never be picked.

**The critical detail:** roads are scored *one at a time* by the same small scoring
function, rather than the officer producing a fixed list like "road 1: 0.3, road 2:
0.5, road 3: 0.2".

A fixed list has a fixed length. A 3-road junction and an 8-road junction would need
different-sized outputs, and a policy trained on one could not handle the other.
Scoring one road at a time means **the same officer handles any junction with any
number of roads.** This is the second half of what makes the policy
topology-agnostic; Step 4 was the first.

---

## Step 7 — Teach the officers

**Thesis: Eq. (9) `eq:loss`**

$$\mathcal{L}=-\mathbb{E}_t[\min(\rho_t\tilde{A}_t,\;
\mathrm{clip}(\rho_t,1{\pm}\varepsilon)\tilde{A}_t)]
+c_v\,\mathbb{E}_t[(V_\phi(s_t)-\hat{V}_t)^2]
-c_e\,\mathbb{E}_t[\mathbb{H}[\pi_\theta]]$$

In words:

```
loss =  (how much better this hop did than expected, but CLIPPED)
      + (how wrong the coach's prediction was)
      − (small bonus for staying open-minded)
```

**The clip** means: do not change your habits too much in one day. If today went
unusually well, do not over-learn from it.

**The coach** is the centralized critic. It sees the entire city and tells each
officer whether a hop was genuinely good or merely lucky. **On deployment day the
coach goes home** — officers then work from local information alone. This is
"centralized training, decentralized execution".

One more detail: **all officers share one brain.** They are not 50 separate learners
but 50 copies of the same officer, so every hop taken anywhere is a lesson for all of
them. That is why 50-node Germany50 does not need 50 times the training data.

---

## How it fits together

| Step | Equation | What it buys you |
|---|---|---|
| Measure | 1, 2 | one number to minimise: the worst road |
| Rules | 3 | loops impossible by construction |
| Score | 4, 5 | feedback every hop, readable score, all cities comparable |
| See | 6 | no names, so unseen cities work |
| Gossip | 7 | local view, wider awareness |
| Choose | 8 | no fixed action list, so any junction size works |
| Learn | 9 | train with a coach, deploy without one |

**One sentence:** the project turns "spread traffic well" into a number (U), makes
that number payable hop by hop so learning is easy, describes junctions so anonymously
that the officer works in any city, and lets officers gossip so they are not blind.

**The thing worth holding onto:** the zero-shot result does not come from the learning
algorithm — it comes from Steps 4 and 6. PPO is standard and the GNN is standard.
Refusing to let the policy see node identities, and refusing to give it a fixed-size
list of actions, is what makes one policy work across seventeen networks and then
three it has never seen.
