# Outline: discretised cross-flow heat transfer around a pipe

This note describes a simple way to model heat transfer between a fluid flowing **inside a pipe** and an external fluid flowing **across the outside of the pipe**.

The pipe is divided into small elements along its length. Each element exchanges heat locally, and the outlet temperature from one element becomes the inlet temperature to the next.

## 1. Physical picture

Consider a pipe carrying an internal fluid. Outside the pipe, another fluid flows across it.

Examples include:

- air flowing across a tube in a heat exchanger
- exhaust gas flowing across a coolant pipe
- ambient air cooling a hot pipe
- liquid cross-flow over a bank of tubes

The internal stream moves along the pipe axis. The external stream moves across the pipe.

```text
External cross-flow
        ↓   ↓
        ↓   ↓
        ↓   ↓
  ┌───────────────────┐
  │-> internal flow → │
  └───────────────────┘
        ↓   ↓
        ↓   ↓
        ↓   ↓
  ```

The pipe wall separates the two fluids, so heat must pass through:

1. the internal convection layer
2. the pipe wall
3. the external convection layer

## 2. Discretising the pipe

Divide the pipe into `N` small elements of length:

$$
\Delta x = \frac{L}{N}
$$

where:

- $L$ = total pipe length
- $N$ = number of pipe elements
- $\Delta x$ = element length

Each element has a heat-transfer area:

$$
\Delta A = \pi D \Delta x
$$

where:

- $D$ = relevant pipe diameter
- usually use outside diameter for external heat-transfer area
- for thin-walled pipes, the difference between inside and outside area is often small

## 3. Local heat-transfer model

For each element, the local heat-transfer rate can be written as:

$$
\Delta \dot{Q} = U \Delta A \left(T_\text{ext} - T_\text{int}\right)
$$

where:

- $\Delta \dot{Q}$ = heat transferred in the element, W
- $U$ = overall heat-transfer coefficient, W/m²/K
- $\Delta A$ = local heat-transfer area, m²
- $T_\text{ext}$ = local external fluid temperature
- $T_\text{int}$ = local internal fluid temperature

The sign convention depends on which direction heat flows. If $T_\text{ext} > T_\text{int}$, heat flows into the pipe. If $T_\text{int} > T_\text{ext}$, heat flows out of the pipe.

## 4. Overall heat-transfer coefficient

The overall resistance is usually modelled as a series of thermal resistances:

$$
\frac{1}{UA}
=
\frac{1}{h_i A_i}
+
R_\text{wall}
+
\frac{1}{h_o A_o}
$$

where:

- $h_i$ = internal convective heat-transfer coefficient
- $h_o$ = external cross-flow convective heat-transfer coefficient
- $A_i$ = internal pipe surface area
- $A_o$ = external pipe surface area
- $R_\text{wall}$ = thermal resistance of the pipe wall

For a cylindrical wall:

$$
R_\text{wall}
=
\frac{\ln \left(r_o / r_i\right)}{2 \pi k L}
$$

where:

- $r_i$ = inner pipe radius
- $r_o$ = outer pipe radius
- $k$ = pipe wall thermal conductivity
- $L$ = pipe length

For one small element, replace $L$ with $\Delta x$.

If the wall is thin and highly conductive, wall resistance is often neglected, giving:

$$
\frac{1}{U}
\approx
\frac{1}{h_i}
+
\frac{1}{h_o}
$$

This approximation assumes the inside and outside areas are nearly the same.

## 5. Internal temperature update

The heat gained or lost by the internal stream is:

$$
\Delta \dot{Q}
=
\dot{m}_i c_{p,i}
\left(
T_{i,n+1} - T_{i,n}
\right)
$$

So the internal outlet temperature from element $n$ is:

$$
T_{i,n+1}
=
T_{i,n}
+
\frac{\Delta \dot{Q}}{\dot{m}_i c_{p,i}}
$$

where:

- $T_{i,n}$ = internal fluid temperature entering element $n$
- $T_{i,n+1}$ = internal fluid temperature leaving element $n$
- $\dot{m}_i$ = internal mass-flow rate
- $c_{p,i}$ = internal specific heat capacity

## 6. External temperature update

If the external fluid temperature is assumed constant, then:

$$
T_{o,n+1} = T_{o,n}
$$

This is a common simplification when the external stream is large compared with the internal stream.

If the external stream temperature also changes, then:

$$
T_{o,n+1}
=
T_{o,n}
-
\frac{\Delta \dot{Q}}{\dot{m}_o c_{p,o}}
$$

where:

- $\dot{m}_o$ = external mass-flow rate interacting with the element
- $c_{p,o}$ = external specific heat capacity

The minus sign appears because heat gained by one stream is lost by the other.

## 7. Eta-M-star form for each element

Instead of using a direct explicit heat-flux update, each element can be treated using an effectiveness-like factor.

Define the local thermal capacity rate of the internal stream:

$$
C_i = \dot{m}_i c_{p,i}
$$

and the local element strength:

$$
M_i^*
=
\frac{U \Delta A}{C_i}
$$

For an internal stream exchanging heat with an external fluid at approximately fixed temperature:

$$
\eta_i = 1 - e^{-M_i^*}
$$

Then the internal fluid update is:

$$
T_{i,n+1}
=
T_{i,n}
+
\eta_i
\left(
T_{o,n} - T_{i,n}
\right)
$$

This form is numerically convenient because it naturally limits the outlet temperature from overshooting the external temperature when the external temperature is fixed.

## 8. If both streams change temperature

If both streams have finite heat capacity, define:

$$
C_i = \dot{m}_i c_{p,i}
$$

$$
C_o = \dot{m}_o c_{p,o}
$$

and:

$$
C_\text{min} = \min(C_i, C_o)
$$

The maximum possible heat transfer in the element is:

$$
\Delta \dot{Q}_\text{max}
=
C_\text{min}
\left(
T_{o,n} - T_{i,n}
\right)
$$

Then use an element effectiveness:

$$
\Delta \dot{Q}
=
\eta \Delta \dot{Q}_\text{max}
$$

The temperature updates are:

$$
T_{i,n+1}
=
T_{i,n}
+
\frac{\Delta \dot{Q}}{C_i}
$$

$$
T_{o,n+1}
=
T_{o,n}
-
\frac{\Delta \dot{Q}}{C_o}
$$

This is closer to a standard heat-exchanger effectiveness model.

## 9. External cross-flow heat-transfer coefficient

The external heat-transfer coefficient $h_o$ depends on the cross-flow conditions.

A typical workflow is:

1. Calculate Reynolds number:

$$
Re_D = \frac{\rho_o V_o D}{\mu_o}
$$

2. Calculate Prandtl number:

$$
Pr = \frac{c_p \mu}{k}
$$

3. Use a cross-flow correlation to estimate Nusselt number:

$$
Nu_D = f(Re_D, Pr)
$$

4. Calculate the heat-transfer coefficient:

$$
h_o = \frac{Nu_D k_o}{D}
$$

where:

- $\rho_o$ = external fluid density
- $V_o$ = cross-flow velocity
- $D$ = pipe outside diameter
- $\mu_o$ = dynamic viscosity
- $k_o$ = thermal conductivity of external fluid
- $Nu_D$ = Nusselt number

For a simple system model, $h_o$ can also be treated as a calibrated or assumed parameter.

## 10. Internal heat-transfer coefficient

The internal heat-transfer coefficient $h_i$ is usually estimated from pipe-flow correlations.

The same general structure is used:

$$
Re_i = \frac{\rho_i V_i D_i}{\mu_i}
$$

$$
Nu_i = f(Re_i, Pr_i)
$$

$$
h_i = \frac{Nu_i k_i}{D_i}
$$

For laminar internal flow, $Nu_i$ may be nearly constant depending on boundary condition.

For turbulent internal flow, $Nu_i$ depends strongly on Reynolds number.

## 11. Algorithm

A simple discretised model looks like this:

  ```text
  Given:
  pipe geometry
  internal inlet temperature
  external inlet or local temperature
  mass flow rates
  fluid properties
  heat-transfer coefficients or correlations

For each element n = 0 to N-1:

  1. calculate local area:
       ΔA = π D Δx

  2. calculate or update h_i and h_o

  3. calculate U

  4. calculate local heat transfer:
       ΔQ = U ΔA (T_o - T_i)

     or use:
       M* = U ΔA / C_i
       η = 1 - exp(-M*)
       T_i,new = T_i + η (T_o - T_i)

  5. update internal temperature

  6. update external temperature if required

  7. pass outlet temperatures to the next element

  ```

  ## 12. Choosing element size

The element size should be small enough that properties and temperatures do not change too much within one element.

A useful rule of thumb is to keep the local value of $M^*$ modest:

$$
M^*_\text{element}
=
\frac{U \Delta A}{\dot{m} c_p}
$$

If $M^*_\text{element}$ is very large, the element may be too long for a simple explicit update.

The exponential eta form is more stable than a simple linear update because it respects the asymptotic approach to the external temperature.

## 13. What this model captures

This model captures:

- axial temperature variation inside the pipe
- changing heat-transfer rate along the pipe
- finite thermal capacity of the internal stream
- optional finite thermal capacity of the external stream
- dependence on flow rate and heat-transfer coefficients
- simple coupling between thermal and flow conditions

## 14. What this model does not capture well

This model may not capture:

- detailed external wake behaviour
- circumferential temperature variation around the pipe
- local separation and recirculation
- non-uniform external velocity fields
- radiation, unless added separately
- wall thermal mass, unless transient wall energy storage is included
- detailed tube-bank effects unless correlations account for them

## 15. Simple interpretation

The discretised cross-flow pipe model treats the pipe as a chain of small heat exchangers.

Each element asks:

```text
Given the local internal and external temperatures,
how much heat can move through this small piece of pipe?
```

The answer depends mainly on:

$$
U \Delta A
$$

which describes heat-transfer strength, and:

$$
\dot{m} c_p
$$

which describes how hard it is to change the fluid temperature.

Their ratio gives a local eta-M-star style measure:

$$
M^*
=
\frac{U \Delta A}{\dot{m} c_p}
$$

Large $M^*$ means the local element strongly changes the fluid temperature.

Small $M^*$ means the local element has only a weak effect.
