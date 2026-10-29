# Conditional T1 parent-streamline reconstruction

Primary source: Creighton, arXiv:gr-qc/0007050v2, pp6,13–14, equations
14,15,A1,A4,A7. Exact PDF SHA256:
`a1e813967f1f822083b297c24008f7a843fa741e296082a0d4646e6426d80aff`.
Retained outside Git at the campaign `temperature/sources/creighton-v2.pdf`.

The mathematical inputs are exact decimals. Observed source evidence is
the published formula and example regime. Parameter tuples are declared
mathematical model conditions, not measured atmospheric conditions.

A1 defines one-sided acceleration PSD as twice the autocorrelation Fourier
transform per Hz. A4 has covariance `sigma² − cT² |r|^p/2`, with Taylor flow
and `0<p<2`. At positive frequency the constant covariance is DC-only.
The one-sided factor2 cancels the covariance divisor2. With
`nu=(p+1)/2`, analytic continuation of the temporal Fourier transform gives
`−2 sqrt(pi) 2^nu/Gamma(−p/2)`, while the transverse correlation-bundle
measure is `2 pi b db v dτ`. Its radial integral contributes
`2 pi 2^nu Gamma(nu+1)`. Their product is
`C(p)=4 pi Gamma(p+2) sin(pi p/2)`.
Reflection and duplication identities establish that coefficient; an
independent high-precision test checks several exponents, including near
the edges. This transformation assumes the paper's smooth large-scale
cutoff and high-frequency continuation, not a full finite atmospheric
covariance model.

The streamline Fourier kernel is
`Gtilde=−2 omega/v² [i cos(theta) K0(omega r/v)
 + sin(theta) cos(psi) K1(omega r/v)]`.
Uniform flow gives `Ftilde=v^(p+3) Gtilde`, and `w=v`.
The real/imaginary cross term vanishes in `|Gtilde|²`. The projected
horizontal acceleration has half-plane reference measure `r dr dpsi`,
`psi∈[0,pi]`, and no streamlines with `r<rmin`. Thus, setting
`x=omega rmin/v` and `In=∫x∞ t Kn(t)² dt`, the full G-free PSD is

`16 pi² Gamma(p+2) sin(pi p/2) (rho/T)² cT² v^(p+2)
 omega^(−p−3) [cos²(theta) I0 + sin²(theta) I1/2]`.

This is kg² m⁻⁴ Hz⁻¹. Multiplying exact conventional `G=6.6743e-11`
squared gives SI acceleration PSD. Detector displacement, arm length,
response, and multi-test-mass summation are excluded. Printed Eq16 has
negative Bessel differences and printed A2/A3 has a different one-sided
normalization; neither is used to set this normalization. Those source
discrepancies require independent science review rather than a fitted
prefactor.

The generator uses the positive double Laplace representation of the
radial square. Writing `a=cosh(u)+cosh(v)`, radial integration from x to
R=x+60 gives `exp(−a x)(x/a+1/a²)−exp(−a R)(R/a+1/a²)`.
Both endpoints are included. Positive tensor Gauss16/32/64 rules on
each fixed panel `0,.125,.25,.5,1,2,4,8` integrate u and v, for each
Bessel order0/1. Physical prefactors, frequency/x and angular quadrature
are recomputed at50/70digits. Arbitrary precision carries the final
`exp(−2x)` together with the observable prefactor so x400 never forces
premature binary64 underflow. No production special function or closed
Bessel difference is used. The slower arbitrary-precision unscaled K
radial integration is retained as a disjoint blind calibration check.

The finite radial endpoint is `R=x+60`. From the Laplace representation
`Kn(t)=∫0∞ exp(−t cosh(u)) cosh(nu) du`, `cosh(u)≥1+u²/2`, and
`cosh(u)≤exp(u²/2)`, one obtains `K0≤exp(−t)sqrt(pi/(2t))` and
`K1≤exp(−t)sqrt(pi/(2(t−1)))` for `t>1`. The second inequality follows
because `d logcosh(u)/du=tanh(u)≤u` for nonnegative u.
Consequently the common radial omitted-tail bound is
`pi/4 R/(R−1) exp(−2R)`. The full observable prefactor and nonnegative
angular weights propagate it into PSD units. This analytic tail bound
does not make the remaining quadrature/roundoff estimates rigorous.

The positive Laplace integrals have finite u/v upper endpoint8. Each
omitted kernel tail satisfies
`L(t)≤exp(−t cosh(8)+8)/(t sinh(8)−1)` from convexity of
`t cosh(u)−u`. Uniform bounds at x are
`K_n(x)exp(x)≤sqrt(2pi/x)exp(1/(2x))` and
`L(x)exp(x)≤exp(−x(cosh(8)−1)+8)/(x sinh(8)−1)`.
The K bound follows by `cosh(nu)≤exp(u)` and extending the completed
Gaussian to the whole real line. Both K and L decay at least
`exp(−(t−x))` for t≥x. Thus the square lost by truncating both Laplace
variables is bounded by
`exp(−2x)(2 Kscaled Lscaled+Lscaled²)(x/2+1/4)`.
This common bound applies to both orders and is propagated with the
same nonnegative angular weights. The tail uncertainty includes the
radial and Laplace omitted domains once each. The smallest planned AUX
x≈.628 ensures the tail denominator is strictly positive.

As x tends to infinity, `Kn~sqrt(pi/(2x))exp(−x)`, so each
`In~pi/4 exp(−2x)`. This predicts the exponential cutoff and the angular
factor `cos²(theta)+sin²(theta)/2`. It is a model asymptotic diagnostic;
it is not a bound on the parent approximation error.

The source p6 gives x about30 for v10m/s,rmin5m,f≥10Hz and x about10
for gale v30m/s. The proposed numerical lower domain is x10, restricted
to finite declared physical tuples. The actual smallest physical x is
`10pi/3`, not an asserted exact10. The parent narrow-bundle assumption,
smooth cutoff, and inertial-range structure statistics remain assumptions.
Their physical approximation error is unquantified, excluded from
numerical oracle U, and never set to zero. Conditional published-model
acceptance must not be called full-atmosphere validation.

Below-domain tests are mathematical AUX. The x400 amplified-prefactor
fixture is a mandatory separate numerical AUX even though x≥10: finite
inputs/intermediates and representable nonzero binary64 G-free/SI PSD are
required. Physical acceptance, numerical AUX status, and delivery readiness
must remain separate; AUX failure blocks delivery without an inferred waiver.
