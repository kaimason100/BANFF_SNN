# Principal BANFF bias-eligibility rule

This document is the canonical code-side definition of the learning rule used
by the publication-ready package. The principal method is **hard-spike-gated,
event-time-aligned local bias eligibility**. A continuous triangular surrogate
is retained only as an optional ablation.

## 1. Forward ALIF state

For one neuron, ignoring the recurrent/input construction around it, the
implemented smooth membrane map is

\[
u_k = E_L + \alpha(u_{k-1}-E_L)
      + (1-\alpha)(I_k+B-w_{k-1}),
\]

with adaptation decay factor \(\beta\). A hard spike is detected from the
full-step candidate voltage. The within-step event fraction \(\rho\in[0,1]\)
is obtained by linear interpolation between the start voltage and that
candidate. The state is then propagated with fractional exponential factors
before and after the event. \(\rho\) is treated as a forward-only,
stop-gradient timing variable.

## 2. Full local bias sensitivity

Define

\[
\epsilon^u = \frac{\partial u}{\partial B},\qquad
\epsilon^w = \frac{\partial w}{\partial B}.
\]

The key point is that adaptation enters the membrane equation as a subtractive
current. Therefore the voltage sensitivity must carry the cross-state term
\(\partial u/\partial w\) forward. Over a smooth interval with membrane decay
factor \(a\),

\[
\epsilon^u_{new}
= a\epsilon^u_{old} + (1-a)(1-\epsilon^w_{old}).
\]

This is the correction relative to the historical implementation that carried
a direct/LIF-like voltage sensitivity and applied the adaptation contribution
only as a temporary correction.

## 3. Propagation to a real LSTI event

For a spike at fraction \(\rho\), the sensitivity is first propagated to the
same event time:

\[
\epsilon^{u,-}_k
= \alpha^{\rho}\epsilon^u_{k-1}
+ (1-\alpha^{\rho})(1-\epsilon^w_{k-1}),
\]

\[
\epsilon^{w,-}_k
= \beta^{\rho}\epsilon^w_{k-1}.
\]

The principal local spike eligibility is

\[
e^{raw}_k = s_k\,\phi\,\epsilon^{u,-}_k,
\]

where \(s_k\in\{0,1\}\) is the actual hard spike and the default explicit event
gain is

\[
\phi=1\;\mathrm{mV}^{-1}.
\]

Thus a non-spiking timestep injects no new eligibility event.

## 4. Reset and adaptation sensitivity

The selected hard-reset branch is treated as stop-gradient:

\[
\epsilon^{u,+}_k=(1-s_k)\epsilon^{u,-}_k.
\]

The spike-triggered adaptation increment is differentiated with the same local
spike eligibility:

\[
\epsilon^{w,+}_k
= \epsilon^{w,-}_k + b\,e^{raw}_k.
\]

The sensitivities are then propagated through the remaining fraction of the
timestep:

\[
\epsilon^u_k
= \alpha^{1-\rho}\epsilon^{u,+}_k
+ (1-\alpha^{1-\rho})(1-\epsilon^{w,+}_k),
\]

\[
\epsilon^w_k
= \beta^{1-\rho}\epsilon^{w,+}_k.
\]

On a non-spiking timestep the same equations reduce to one full smooth step
with no event injection.

## 5. Decoder-matched eligibility filter

The instantaneous event eligibility is inserted at the same LSTI fraction
\(\rho\) into a two-stage rise-decay eligibility cascade with the same decay
factors as the filtered spike state used by the fixed decoder. Non-spiking
steps only advance/decay these eligibility states.

The bias-gradient estimator is the three-factor product

\[
\widehat{\nabla_B L}
= \sum_k \left(W_{out}^{\top}g^z_k\right)\odot\bar e_k,
\]

where \(\bar e_k\) is the decoder-matched filtered local eligibility.
Recurrent nonlocal derivatives are not backpropagated through time.

## 6. Optional continuous-surrogate ablation

Setting `eligibility_mode="surrogate"` replaces the hard local event gate by

\[
\psi(u)=A\max\left(0,1-\frac{|u-V_{th}|}{W}\right).
\]

The package defaults for this ablation are `A=0.7 1/mV` and `W=10 mV`; these
are **not** the principal publication rule and should be reported explicitly
whenever this ablation is used. On a spiking step the local state sensitivity
is evaluated at the LSTI pre-event state; on a non-spiking step it is the
end-step sensitivity. The pseudo-derivative itself is evaluated at the
full-step candidate voltage used by the hard spike decision.

## 7. Terminology

The principal method should be described as an **event-gated e-prop-style
three-factor estimator for intrinsic bias plasticity**, not as the canonical
Bellec synaptic ALIF e-prop rule. The learnable parameter is an intrinsic
bias current rather than a synaptic weight, and the neuron uses subtractive
current adaptation rather than an adaptive threshold.
