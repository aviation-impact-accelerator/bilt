# The Zen of Bilt

1. **Systems define the model; implementations provide its relationships. Execution is derived.**
   Systems define meaningful boundaries and values. Implementations supply equations, transforms and subsystems. Bilt determines how to evaluate them.

2. **Describe relationships, not procedures.**
   State what is true between values. Supply what is known, request what is wanted, and let Bilt determine the route between them.

3. **Equations express relationships; transforms express direction.**
   Use equations for genuinely acausal relationships and transforms where direction is intrinsic to the computation.

4. **Keep the structure close to the engineering.**
   Systems, values and equations should reflect the concepts and mathematics of the problem rather than computational convenience.

5. **Compose systems rather than building pipelines.**
   Build larger models from meaningful subsystems. Connections express shared quantities, not the movement of data from one computation to another.

6. **Values carry meaning.**
   A value's identity, units and domain describe what it represents; they are not merely metadata around an array.

7. **Implementations should be replaceable.**
   Different relationships, numerical methods, backends or levels of fidelity should be able to realise the same system.

8. **Prefer explicit structure to hidden behaviour.**
   Avoid hidden state, accidental broadcasting and implementation tricks that obscure the model or make it harder to analyse.

9. **Execution should follow transparently from the model.**
   It should be possible to inspect why Bilt selected a particular set of relationships and how those relationships produced the requested output.

10. **Let the modeller provide meaning; let Bilt handle mechanics.**
    Modellers define systems, relationships and assumptions. Bilt handles structural analysis, equation isolation, broadcasting, lowering and execution.
