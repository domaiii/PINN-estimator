# Quellenunsicherheitsbasierte Gasexploration mit gemeinsamer Last-Layer-Laplace-Approximation

## Ziel: Auswahl des nächsten Messpunkts

Wir betrachten ein Gebiet $\Omega\subset\mathbb R^2$. Das trainierte PINN
liefert darin Schätzungen der Konzentration $c(x)$ und der Quellenstärke
$q(x)$. Beide Netze sind insbesondere über den Residual der stationären
Advektions-Diffusions-Gleichung

$$
-D\Delta c+\mathbf u\cdot\nabla c-q=0
$$

im gemeinsamen Loss gekoppelt. Dabei sind $D>0$ der konstante
Diffusionskoeffizient und $\mathbf u$ das bekannte, festgehaltene
Geschwindigkeitsfeld. Die räumlichen Operatoren $\nabla$ und $\Delta$
bezeichnen Gradient und Laplace-Operator.

Für die aktive Exploration suchen wir den Messpunkt, an dem eine
Konzentrationsmessung die Unsicherheit des Quellenfeldes voraussichtlich
am stärksten reduziert. Sei $\mathcal X\subset\Omega$ eine nichtleere,
endliche Menge erreichbarer Messkandidaten und seien
$r_1,\ldots,r_M\in\Omega$ feste Auswertungspunkte mit $M\geq1$.

Eine zukünftige Konzentrationsmessung bei $x$ modellieren wir als

$$
y_x=c(x)+\eta_x,
\qquad
\eta_x\sim\mathcal N(0,\sigma_{\mathrm{obs}}^2),
$$

mit bekanntem, von den Feldern unabhängigem Messrauschen und
$\sigma_{\mathrm{obs}}^2>0$.

Gesucht ist ein Messpunkt, der die erwartete verbleibende Quellenvarianz
im Mittel über die Auswertungspunkte minimiert:

$$
x_{\mathrm{next}}
\in
\arg\min_{x\in\mathcal X}
\mathbb E_{y_x}\!\left[
\frac1M\sum_{j=1}^{M}
\operatorname{Var}[q(r_j)\mid y_x]
\right].
$$

Alle Varianzen und Kovarianzen beziehen sich auf die Unsicherheit nach
Berücksichtigung der bisherigen Messdaten und der Modellannahmen.
Die dafür benötigte Verteilung der Feldvorhersagen konstruieren wir
weiter unten.

Unter einer gemeinsamen Gaußapproximation gilt

$$
\operatorname{Var}[q(r)\mid y_x]
=
\operatorname{Var}[q(r)]
-
\frac{
\operatorname{Cov}[q(r),c(x)]^2
}{
\operatorname{Var}[c(x)]+\sigma_{\mathrm{obs}}^2
}.
$$

Die verbleibende Varianz hängt in dieser Approximation vom Messort,
aber nicht vom tatsächlich beobachteten Messwert ab. Der Erwartungswert
entfällt daher. Außerdem hängt die bisherige Quellenvarianz
$\operatorname{Var}[q(r)]$ nicht vom Messkandidaten $x$ ab.

Wir können somit statt der verbleibenden Varianz ihre Verringerung
maximieren:

$$
\boxed{
x_{\mathrm{next}}
\in
\arg\max_{x\in\mathcal X}
\frac1M\sum_{j=1}^{M}
\frac{
\operatorname{Cov}[q(r_j),c(x)]^2
}{
\operatorname{Var}[c(x)]+\sigma_{\mathrm{obs}}^2
}.
}
$$

**Zur Berechnung von $x_{\mathrm{next}}$ benötigen wir also zwei Größen:**
die Konzentrationsvarianz $\operatorname{Var}[c(x)]$ und die
Kreuzkovarianz $\operatorname{Cov}[q(r_j),c(x)]$. Das trainierte PINN
liefert zunächst nur feste Feldschätzungen. Deshalb approximieren wir
nun die gemeinsame Parameterunsicherheit beider Netze und übertragen
sie anschließend auf diese beiden Größen.

## Gemeinsame Parameterunsicherheit

### Warum eine gemeinsame Last-Layer-Approximation?

Um die benötigten Varianzen und Kovarianzen zu erhalten, betrachten wir
die Netzparameter als unsicher. Eine Laplace-Approximation für sämtliche
Parameter wäre jedoch aufwendig. Deshalb halten wir nach dem Training
die Hidden Layers fest und betrachten nur die Gewichte und Biases der
letzten affinen Layer als unsicher.

![Aufbau der gemeinsamen Last-Layer-Laplace-Approximation](assets/joint_last_layer_laplace_architecture.svg)

Die festen Hidden Layers liefern die Featurevektoren

$$
h_c(x)\in\mathbb R^{m_c},
\qquad
h_q(x)\in\mathbb R^{m_q},
$$

Die Last-Layer-Parameter fassen wir zusammen als

$$
\theta_c=
\begin{bmatrix}w_c\\b_c\end{bmatrix},
\qquad
\theta_q=
\begin{bmatrix}w_q\\b_q\end{bmatrix},
\qquad
\theta=
\begin{bmatrix}\theta_c\\\theta_q\end{bmatrix}.
$$

Hier sind $w_c\in\mathbb R^{m_c}$ und $w_q\in\mathbb R^{m_q}$ die
Gewichtsvektoren sowie $b_c,b_q\in\mathbb R$ die Biases.

**Wir müssen beide Parametersätze gemeinsam betrachten, weil wir für das
Auswahlkriterium ihre Kopplung benötigen.** Zwei getrennte, unabhängige
Approximationen würden die gesuchte Kreuzkovarianz nicht erfassen.

### Wie erhalten wir die Parameterkovarianz?

Sei $L(\theta)$ der gemeinsame Loss bei festgehaltenen Hidden Layers.
Wir nehmen an, dass er bis auf eine additive Konstante einem negativen
Log-Posterior entspricht. Seine Skalierung und Gewichtung legen damit
auch die angenommene Unsicherheit fest.

In der Nähe des trainierten Optimums $\theta^\ast$ mit
$\nabla_\theta L(\theta^\ast)\approx0$ gilt

$$
L(\theta)
\approx
L(\theta^\ast)
+
\frac12(\theta-\theta^\ast)^\top
H(\theta-\theta^\ast),
\qquad
H=\nabla_\theta^2L(\theta^\ast).
$$

Diese quadratische Näherung liefert bei positiv definitem $H$ die
gaußsche Parameterapproximation

$$
\theta\sim\mathcal N(\theta^\ast,\Sigma),
\qquad
\Sigma=H^{-1}.
$$

Die gemeinsame Kovarianz hat die Blockstruktur

$$
\Sigma=
\begin{bmatrix}
\Sigma_{cc}&\Sigma_{cq}\\
\Sigma_{qc}&\Sigma_{qq}
\end{bmatrix},
\qquad
\Sigma_{cq}=\Sigma_{qc}^\top.
$$

Für unser Auswahlkriterium benötigen wir insbesondere
$\Sigma_{cc}$, die Kovarianz der Konzentrationsparameter, und
$\Sigma_{qc}$, die Kreuzkovarianz zwischen Quellen- und
Konzentrationsparametern.

> **Numerische Stabilisierung**
>
> In der Umsetzung symmetrisieren wir die berechnete Hesse-Matrix:
>
> $$
> H_s=\frac12(H+H^\top)
> =U\operatorname{diag}(\lambda_i)U^\top.
> $$
>
> Dabei sind $\lambda_i$ die Eigenwerte und die Spalten von $U$
> orthonormale Eigenvektoren. Mit $\varepsilon=10^{-7}$ verwenden wir
>
> $$
> \Sigma=
> U\operatorname{diag}\!\left(
> \frac1{\max(\lambda_i,\varepsilon)}
> \right)U^\top.
> $$
>
> $\Sigma$ ist damit die Inverse einer positiv definiten Ersatzmatrix.
> Werden Eigenwerte begrenzt, verändert dies die Approximation;
> insbesondere sind deutlich negative Eigenwerte kein reines
> Rundungsproblem.

## Konzentrationsvarianz am Messkandidaten

Für den Nenner des Auswahlkriteriums benötigen wir
$\operatorname{Var}[c(x)]$. Dazu übertragen wir die Parameterkovarianz
$\Sigma_{cc}$ auf den Netzausgang am Ort $x$.

Die trainierte Vorhersage $c(x;\theta_c^\ast)$ ist fest. Die unsichere
Vorhersage $c(x)=c(x;\theta_c)$ entsteht durch die approximierte
Parameterverteilung. Mit

$$
\delta\theta_c=\theta_c-\theta_c^\ast,
\qquad
g_c(x)=
\left.
\nabla_{\theta_c}c(x;\theta_c)
\right|_{\theta_c=\theta_c^\ast}
$$

linearisieren wir den Ausgang:

$$
c(x)
\approx
c(x;\theta_c^\ast)+g_c(x)^\top\delta\theta_c.
$$

Da der erste Term konstant ist, folgt unmittelbar

$$
\boxed{
\operatorname{Var}[c(x)]
\approx
g_c(x)^\top\Sigma_{cc}g_c(x).
}
$$

Zur Berechnung von $g_c(x)$ schreiben wir den Netzausgang als

$$
c(x;\theta_c)
=
\operatorname{softplus}\!\left(
\theta_c^\top\widetilde h_c(x)
\right),
\qquad
\widetilde h_c(x)=
\begin{bmatrix}h_c(x)\\1\end{bmatrix}.
$$

Die angehängte Eins berücksichtigt den Bias.
Mit $\operatorname{softplus}(z)=\log(1+e^z)$ und ihrer Ableitung
$\operatorname{sigmoid}(z)=1/(1+e^{-z})$ ergibt die Kettenregel

$$
g_c(x)
=
\operatorname{sigmoid}\!\left(
(\theta_c^\ast)^\top\widetilde h_c(x)
\right)\widetilde h_c(x).
$$

Damit ist die benötigte Konzentrationsvarianz berechenbar.

## Kopplung zwischen Quellenstärke und Konzentration

Für den Zähler fehlt noch $\operatorname{Cov}[q(r),c(x)]$. Dazu
linearisieren wir auch den Ausgang des Quellennetzes:

$$
q(r)
\approx
q(r;\theta_q^\ast)+g_q(r)^\top\delta\theta_q,
\qquad
\delta\theta_q=\theta_q-\theta_q^\ast.
$$

Das Quellennetz verwendet ebenfalls einen Softplus-Ausgang. Daher gilt
analog zur Konzentration

$$
\widetilde h_q(r)=
\begin{bmatrix}h_q(r)\\1\end{bmatrix},
\qquad
g_q(r)=
\left.
\nabla_{\theta_q}q(r;\theta_q)
\right|_{\theta_q=\theta_q^\ast}
=
\operatorname{sigmoid}\!\left(
(\theta_q^\ast)^\top\widetilde h_q(r)
\right)\widetilde h_q(r).
$$

Aus der gemeinsamen Parameterapproximation kennen wir bereits

$$
\operatorname{Cov}[\delta\theta_q,\delta\theta_c]=\Sigma_{qc}.
$$

Mit den beiden Linearisierungen folgt deshalb

$$
\boxed{
\operatorname{Cov}[q(r),c(x)]
\approx
g_q(r)^\top\Sigma_{qc}g_c(x).
}
$$

Die Gradienten übertragen die Parameterkopplung auf die konkreten
Orte $r$ und $x$.

Zugleich begründen die Linearisierungen die eingangs verwendete
Gaußapproximation: Die linearisierten Felder sind gemeinsam
gaußverteilt, weil sie affine Funktionen der gemeinsam gaußverteilten
Parameter sind. Für die ursprünglichen nichtlinearen Softplus-Ausgänge
gilt dies nur näherungsweise.

## Berechnung des nächsten Messpunkts

Nun sind beide Größen des Auswahlkriteriums bekannt. Einsetzen ergibt

$$
\boxed{
s(x)
=
\frac{
\frac1M\sum_{j=1}^{M}
\left[
g_q(r_j)^\top\Sigma_{qc}g_c(x)
\right]^2
}{
g_c(x)^\top\Sigma_{cc}g_c(x)
+\sigma_{\mathrm{obs}}^2
},
\qquad
x_{\mathrm{next}}
\in
\arg\max_{x\in\mathcal X}s(x).
}
$$

Praktisch bedeutet dies:

1. Die Hidden Layers und den Wind festhalten und aus dem gemeinsamen
   Loss die stabilisierte Last-Layer-Kovarianz $\Sigma$ berechnen.
2. Die Gradienten $g_q(r_j)$ an den festen Auswertungspunkten und
   $g_c(x)$ an den Messkandidaten bestimmen.
3. $s(x)$ auswerten und einen erreichbaren Kandidaten mit maximalem
   Wert auswählen.

Der rauschfreie Spezialfall ergibt sich mit
$\sigma_{\mathrm{obs}}^2=0$, sofern der Nenner positiv ist.

Der Auswahlwert beschreibt die prognostizierte Verringerung der
mittleren Quellenvarianz an den Auswertungspunkten im aktuellen
linearisierten Modell. Er erfasst nur die Last-Layer-Unsicherheit und
garantiert nicht dieselbe Varianzreduktion nach erneutem Training des
gesamten PINN.