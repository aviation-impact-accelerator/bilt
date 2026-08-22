# The Teachings

These are invented dialogues in the style of the Zen kōan tradition, exploring Bilt through the Buddhist ideas of dependent arising and emptiness.

Nothing in the model exists independently.

The execution is the web, not the chain.

<details>
<summary><em>For the engineer</em></summary>

**Pratītyasamutpāda** (dependent co-arising): nothing exists on its own. Every phenomenon arises in dependence on others.

In Bilt, values exist through their relationships to other values. Structural analysis asks which relationships can connect what is known to what is requested. Only then is an order of execution derived.

**Śūnyatā** (emptiness): no object has an inherent, independent essence. What we call a "thing" is understood through its relations.

In Bilt, numerical representation and model meaning are distinct. An array is the representation. A value carries semantic identity.

**The Middle Way** (*Madhyamaka*): the truth is not that something exists independently, nor that it does not exist at all.

In Bilt, the model is acausal in what it declares and causal in how a particular question is answered.

Bilt declares the web.

Evaluation finds the route.

</details>

<details>
<summary><em>For the functional programmer</em></summary>

Functional programming begins from a similar instinct: describe computation through values and composition rather than histories of mutation.

Bilt carries that instinct further.

The modeller declares relationships.

The route through them may itself be derived.

</details>

The koans disrupt.

The annotations ground.

Read the koan. Open the annotation if you need it. Then move on.

---

A student asked:

"Where is the execution?"

The master said:

"You wrote the model.

The execution was already in it.

You were looking for a second thing.

There is no second thing."

A student asked:

"What is the execution?"

The master said:

"You."

The student said:

"But I wrote the model."

The master said:

"And then you asked it a question.

Without the model there is no route.

Without the question there is no route.

You are looking for the route as though it existed before either.

There is no route before the walking."

<details>
<summary><em>For the engineer</em></summary>

The execution is not separately authored after the model.

The model declares possible relationships. An evaluation supplies a question: these values are known; this value is wanted.

From those together, Bilt derives a route.

There is no second procedural model hidden behind the declarative one.

</details>

---

A student asked:

"What is an equation?"

The master said:

"It has no before.

It has no after.

You gave it two truths.

It does not ask which came first.

This is the acausal.

Do not mistake it for the causal."

A student asked:

"What is a transform?"

The master said:

"It has a before.

It has an after.

This becomes that.

Do not mistake it for the acausal."

A student asked:

"An equation has no direction.

How does Bilt know which way to solve it?"

The master said:

"If the direction lived in the equation,

the equation was never acausal.

If no direction could ever be chosen,

the equation could produce nothing.

You ask the equation for its direction.

Ask instead what is known.

Ask what is wanted."

<details>
<summary><em>For the engineer</em></summary>

An equation declares a relation, not a direction.

Structural analysis chooses a direction from the values available and the value requested. The direction belongs to the evaluation, not to the equation.

A transform is different: its direction is part of its declaration.

The model may therefore be acausal in what it says and causal in how a particular question is answered.

</details>

<details>
<summary><em>For the functional programmer</em></summary>

Functional programming describes values through relationships rather than sequences of mutation.

A transform says:

`from this, derive that`

An equation goes further:

`these must agree`

The direction comes later.

</details>

---

A student asked:

"How do systems connect?"

The master said:

"Two systems meet.

They do not pass anything between them.

They share a value.

This is not a handoff.

This is a meeting.

A pipeline passes a tensor.

A system shares a value.

Which is the teaching?"

<details>
<summary><em>For the engineer</em></summary>

In a pipeline, data moves from producer to consumer.

In a system model, connected components participate in the same relationship. One does not have to manufacture a value merely so that another can receive it.

The connection belongs to the model, not to an execution order.

Direction may be derived later.

</details>

---

A student asked:

"What is a value?"

The master said:

"You call it an array.

Look again.

The array is the carrier.

The value is the meaning.

A value has a name.

A quantity has a unit.

A tensor has a domain.

The array is what you compute.

The value is what you mean."

A student asked:

"The value is the meaning.

The array is the carrier.

When Bilt computes, what is it computing?"

The master said:

"Meaning is not computed.

It is declared.

The array takes the shape required by the meaning.

You declared the relationships.

Bilt carried them into computation.

Do not ask the array what it means.

It does not know."

<details>
<summary><em>For the engineer</em></summary>

The array is numerical representation. The Bilt value is semantic identity.

Name, units and domain are not discovered by inspecting the array. They come from the model.

Computation changes representation.

It does not invent meaning.

</details>

---

A student asked:

"How many implementations does a system have?"

The master said:

"One.

Many.

Neither.

The system is the question.

The implementation is an answer.

The system does not know its answer.

It does not need to."

A student asked:

"The system does not know its implementation.

Then how does it constrain which implementations are valid?"

The master said:

"Show me the implementation without the system."

The student could not.

The master said:

"Now show me the system without choosing an implementation."

The student did.

The master said:

"Good.

The question may exist before its answer.

But an answer is only an answer to something."

<details>
<summary><em>For the engineer</em></summary>

The system defines the interface: the values that exist and the questions that may be asked of them.

An implementation supplies one set of relationships capable of answering those questions.

A system need not know which implementation will be used.

An implementation only has meaning as an implementation of a system.

</details>

---

A student asked:

"Is there hidden state in the execution?"

The master said:

"What is not in the model

must still have a reason.

No unexplained step.

No secret movement.

You are looking for a mystery.

There should be none."

<details>
<summary><em>For the engineer</em></summary>

Every step in an execution should be traceable either to something declared in the model or to an explicit consequence of it.

Equation isolation, domain alignment, broadcasting, structural analysis and lowering may happen automatically.

Automatic does not mean unexplained.

Given the model, implementation, known values and requested outputs, Bilt should be able to explain how the result was obtained.

</details>

<details>
<summary><em>For the functional programmer</em></summary>

Pure functions have no secret history.

Their meaning follows from their inputs and relationships, not from hidden mutation elsewhere.

Bilt seeks the same property at the modelling level: sophisticated mechanics, but no unexplained semantics.

</details>

---

A student asked:

"Why did Bilt choose this plan?"

The master said:

"Ask the plan.

It will answer.

It has no reason it cannot give.

The route is not a mystery.

It is a consequence."

A student asked:

"The plan is inspectable.

So the plan is separate from the execution."

The master said:

"Separate from what?"

The student said:

"The model."

The master said:

"Change what is known.

Does the plan remain?"

"No."

"Change what is wanted.

Does the plan remain?"

"No."

"Change the model.

Does the plan remain?"

"No."

The master said:

"Then why do you speak of it as though it stood alone?"

The student said:

"Because I can inspect it."

The master said:

"You can inspect a shadow.

That does not make it independent of what casts it."

<details>
<summary><em>For the engineer</em></summary>

The plan is structural analysis made explicit: which relationships are used, in which direction, across which systems, and in what order.

It is not a second model.

It is a derived view of this model answering this particular question.

Because it is data, it can be inspected before numerical execution.

Change the model, implementation, known values or requested output, and the plan may change.

The plan explains the execution because the execution follows the plan.

</details>

---

A student asked:

"If the plan can be inspected before execution,

has Bilt done anything yet?"

The master said:

"Does a path exist before you walk it?"

The student said:

"Yes."

The master said:

"Did it exist before you chose where to go?"

The student said nothing.

The master said:

"Now you understand compilation."

<details>
<summary><em>For the engineer</em></summary>

Structural analysis and numerical execution are different stages.

Bilt can determine *how* a result will be obtained before performing the numerical work.

The plan can therefore be inspected before execution without being independently authored.

</details>

---

A student asked:

"What is my role?"

The master said:

"You built the model.

Bilt walked it.

You did not walk it.

Bilt did not build it.

This is the division.

This is the teaching.

Now go."

<details>
<summary><em>For the engineer</em></summary>

You define the systems, values, equations and transforms.

Bilt determines how the requested result can be obtained: structural analysis, equation isolation, alignment, lowering and execution.

You own the meaning.

Bilt owns the mechanics.

</details>

<details>
<summary><em>For the functional programmer</em></summary>

Functional programming separates meaning from much of the machinery of execution. Compilers exploit that freedom to transform computation while preserving semantics.

Bilt aims for the same separation in engineering models.

The modeller states what the relationships mean.

Bilt determines how a requested result can be obtained.

Imperative programming describes the walking.

Functional programming describes the path.

Bilt describes the landscape and asks the compiler to find a path.

</details>

---

*The model is declared.*

*The question is asked.*

*Execution is derived.*

*Now go.*

---

*Inspired by the Buddhist ideas of dependent arising and Madhyamaka, particularly the work of Nāgārjuna, and by the functional-programming traditions of immutability, composition and separation of semantics from execution.*
