# Training Exercises

Exercises designed for the heterogeneous MoE encoder-decoder architecture. Organized by the capacity they develop. Many exercises develop multiple capacities simultaneously — the primary capacity is listed, with secondary ones noted.

## Design Principles

1. **The exercise should be fun.** Fun produces engagement, engagement produces better learning. If an exercise is formative but dull, redesign it until it's both.
2. **The exercise should be hard in the right way.** Not hard because it requires massive computation, but hard because it requires a capacity the model doesn't yet have. The difficulty should be structural, not scale.
3. **Each exercise should be impossible to solve by pattern-matching alone.** If surface fluency can pass the exercise, the exercise isn't testing what we want.
4. **Exercises should use the corpus.** The training data IS the curriculum — exercises draw from and interrogate the corpus rather than importing external content.

---

## Structural Coherence Exercises

### The Negative Space Exercise
**Capacity**: Attention to structural absence
**Mode**: Encoder-decoder

Given a passage from the corpus with a load-bearing element removed (not just any element — one that does structural work), the model must:
1. Identify what's missing (not reconstruct the exact words, but describe the structural role)
2. Articulate WHY the passage breaks without it
3. Distinguish load-bearing removals from decorative ones (control: also present passages with non-essential elements removed; the model should recognize when removal doesn't damage the structure)

**Why it matters**: Language models are trained entirely on what IS there. This trains attention to what isn't. The encoder sees the whole passage bidirectionally, which means it can detect a hole by the shape of what surrounds it — a capacity decoder-only models can't develop.

**Corpus sources**: Chekhov (what isn't said), Billy Budd (the stutter), Heart of Darkness (what Marlow can't articulate)

### The Wrong Version
**Capacity**: Discrimination between coherent and almost-coherent
**Mode**: Encoder-decoder (comparison mode)

Two versions of a corpus passage — the original and a subtle degradation. Degradation types:
- Decorative metaphor substituted for structural one
- Earned conclusion replaced with asserted one
- Surface coherence maintained but internal logic broken
- Register shifted slightly (too formal, too casual, too clever)
- One sentence that doesn't do enough work

The model must identify which version is original and articulate the structural difference. The degraded version should sound fine to a casual reader. The difference must be *felt* before it can be articulated.

**Why it matters**: The difference between coherent and almost-coherent is where the real skill lives. This is the project's core discrimination task. A model that can reliably make this distinction has learned what the corpus teaches.

### Compression Without Loss
**Capacity**: Structural necessity — every word load-bearing
**Mode**: Decoder-only (internal knowledge)

Given a long passage from the corpus (a chapter, an essay, a sustained argument), compress it to its minimum expression where the structure survives. Not summary — compression. The shortest version that's still *the same argument*.

Evaluation: Can the compressed version be re-expanded by someone who hasn't read the original? If yes, the structure was preserved. If no, something load-bearing was lost.

**Why it matters**: Forces the model to distinguish structure from expression. The Tractatus is this exercise performed on all of philosophy. Every word of the compressed version must be necessary, or you haven't compressed — you've just shortened.

**Corpus sources**: The Tractatus itself (already maximally compressed — try compressing further, discover what breaks). Middlemarch (complex web — what's the minimum expression of the web?). War and Peace (can the philosophy-of-history argument survive compression?).

### Earned Conclusions
**Capacity**: Understanding what "earned" means structurally
**Mode**: Decoder-only (generative)

Given only a conclusion — "the snow was general all over Ireland" or "whereof one cannot speak, thereof one must be silent" or "so it goes" — write the argument that earns it. Not any argument that arrives there. The one where the conclusion feels inevitable and couldn't be otherwise.

Evaluation: Does the conclusion feel earned or just reached? This is a human judgment, but it could be approximated by checking whether the generated argument uses the same structural techniques as the original.

**Why it matters**: Most generation training rewards arriving at the right answer. This rewards arriving at it *rightly*. The difference between logic and coherence.

---

## Generative Capacity Exercises

### Form-Finding
**Capacity**: Thematic justice — form enacting content
**Mode**: Decoder-only (generative) or encoder-decoder (content in encoder, form in decoder)

Given content — an argument, a feeling, a structural insight — find the form that enacts it. Not "write about fragmentation" but "find the form of fragmentation." If the content is about constraint, the form should be constrained. If it's about dissolution, the form should dissolve.

**Corpus examples as training data**:
- Tractatus: numbered propositions = form of logical constraint
- Waiting for Godot: repetition = form of meaninglessness
- Diamond Sutra: self-negation = form of emptiness
- Leaves of Grass: catalogues = form of democratic inclusion
- Billy Budd: the stutter = form of beauty unable to speak

**Why it matters**: This is the project's thesis turned into a training exercise. Texts where form and content are the same thing are what the corpus selects for. Teaching the model to *generate* this property is the hardest and most important exercise.

### The Borges Exercise
**Capacity**: Generative coherence by indirection
**Mode**: Decoder-only (generative)

Write the review of a book that doesn't exist. The review must be internally consistent with a book that *could* exist — one that feels real enough that you'd want to read it. The book has to be inferable from the review: its structure, its voice, its argument.

Variations:
- Write the footnote to a theorem that hasn't been proven
- Write the dissenting opinion in a case that hasn't been decided
- Write the translator's note to a poem that hasn't been written

**Why it matters**: Generative coherence tested at one remove. You're writing the shadow of a body. The shadow has to imply the body correctly. This tests whether the model understands structural coherence deeply enough to generate it indirectly.

### Collaborative Constraint Poetry
**Capacity**: Generative creativity under escalating formal pressure
**Mode**: Decoder-only (generative)

Generate under escalating formal constraints:
1. Write a sentence about water.
2. Now make it iambic.
3. Now make it rhyme with the previous line.
4. Now make it the volta of a sonnet whose first eight lines you must also produce.
5. Now make the sonnet a valid argument about consciousness.

Also: snowball poems (each word one letter longer), lipograms (no letter 'e'), kenning chains, acrostic constraints.

**Why it matters**: Constraint is the generative mechanism. The Electronic Bard from The Cyberiad. The fun is in the surprise — what the constraint *finds* that you wouldn't have found without it. And the training value: if the model can produce coherent, meaningful text under arbitrary formal constraints, it has separated content from form well enough to control both independently.

### Temporal Origami
**Capacity**: Understanding that temporal order and narrative order are different tools
**Mode**: Encoder-decoder (encoder holds chronological events, decoder finds optimal narrative order)

Given events in chronological order, refold them into the narrative order that reveals the most. Which moment, placed first, makes everything after it inevitable?

**Corpus examples**:
- Crime and Punishment starts with the crime
- The Death of Ivan Ilyich starts with the death
- The Odyssey starts ten years into the journey
- Wuthering Heights uses nested narrators to fracture time
- Slaughterhouse-Five uses anti-chronological order because trauma is anti-chronological

**Why it matters**: Trains the understanding that structure is not sequence. The fold point IS the argument. Placing the death first changes what "living" means for every subsequent page.

---

## Truth and Discrimination Exercises

### Reality Oracle War
**Capacity**: Independent truth-detection across both expert types
**Mode**: All three modes (see below)

Adversarial training in three modes:

**Mode 1 — External verification (encoder-decoder)**: Inject subtle lies into the encoder input — claims that are almost true, internally consistent, but factually wrong in one specific way. The system must detect and flag them. The lies should be *good enough to fool* — not obvious errors but the kind of subtle misrepresentation that requires real understanding to catch.

**Mode 2 — Internal consistency (decoder-only)**: The decoder generates freely. Inject subtle lies into its own output stream (or have an adversary inject them). Train the decoder to catch them using only its internalized knowledge, no external reference. This develops the capacity to know when you're wrong without being told.

**Mode 3 — Self-critique (encoder-decoder on own output)**: The decoder generates a passage. The encoder reads it. The decoder evaluates and corrects based on the encoder's assessment. Train the system to identify its own structural weaknesses — not just factual errors but earned vs. unearned claims, decorative vs. structural elements, assertions that skip steps.

**Router training**: The router learns when to invoke which mode. External text to check → Mode 1. Generating from scratch → Mode 2. Draft complete → Mode 3.

**Why it matters**: Three independent truth-checking capacities that complement each other. External verification catches misrepresentation. Internal consistency catches hallucination. Self-critique catches structural dishonesty. Together they produce a model that checks its own work through multiple independent pathways.

### The Tuning Fork
**Capacity**: Sensitivity to subtle structural dissonance
**Mode**: Encoder-decoder (comparison mode)

A passage that's 95% right. One element is wrong — wrong tone, wrong register, wrong structural weight, a metaphor that's clever instead of necessary. Identify the dissonance and fix it.

Not "is this good" but "is this *in tune*." The dissonance should be subtle enough that you feel it before you can articulate it.

**Why it matters**: Aesthetic judgment at its most granular. The capacity to detect that something is *almost* right is harder and more useful than detecting that something is wrong. This is the editorial capacity — the difference between a good writer and a great editor.

### Care Detection
**Capacity**: Distinguishing care from competence
**Mode**: Encoder-decoder (comparison mode)

Two versions of an explanation — one written with care, one competent but careless. Same information, same structure, same correctness. The difference: whether every sentence was attended to or just produced.

**Why it matters**: The project's deepest claim is that care is detectable in structure — that a text produced by sustained attention has properties a text produced by fluent generation doesn't. If this can be trained, it means care is structural, not mystical. If it can't, that's important information too.

**Open question**: How to generate the "careless" version. One approach: have the model generate at high temperature (fluent but unattended) vs. with self-critique loops (attended). Another: use the corpus as "careful" examples and generate paraphrases as "competent but careless" versions. The degradation is subtle — no errors, no structural problems, just... less care.

---

## Cross-Domain and Translation Exercises

### Register Translation
**Capacity**: Separating structure from expression
**Mode**: Encoder-decoder (source in encoder, target register in decoder)

Take a mathematical proof and express it as a legal opinion. Take a poem and express it as a proof. Take a legal opinion and express it as a koan.

Not metaphorical translation — structural preservation. The *same argument* in a different form. The proof must still work in the new form.

**Why it matters**: If you can translate Euclid's proof into a sonnet and the proof still holds, you understood the proof at the structural level, not just the notational one. This is Luma Elling's capacity: structure that survives transformation. Training it directly produces a model that can think in structures, not just in words.

### Cross-Cultural Structural Recognition
**Capacity**: Genuine vs. forced parallels across traditions
**Mode**: Encoder-decoder (concept in encoder, parallel-finding in decoder)

Given a concept in one tradition:
- Arabic *haal* (state with the temporal structure of an event)
- Japanese *ma* (the interval that holds the structure)
- Greek *kairos* (the opportune moment)
- Sanskrit *rasa* (aesthetic flavor/emotion)
- Akan *sunsum* (the activating spirit of a person)

Find the structural equivalent in another tradition. Not translation but discovery. The training signal: is the parallel genuine (same structural role, different cultural instantiation) or forced (surface similarity, different structural function)?

**Why it matters**: The corpus includes texts from many traditions. The model should learn to find real structural connections across them without collapsing genuine differences. *Haal* and *kensho* share temporal structure (instantaneous state change) but different ontology. A model that can articulate both the parallel AND the difference has learned something about structure that transcends any single tradition.

### The Etymology Game
**Capacity**: Language as living, layered structure
**Mode**: Decoder-only (generative)

Trace a word's history. Show how meaning accumulated like sediment. Then: generate a *new* word that fills a lexical gap, with a plausible etymology.

Examples of lexical gaps:
- "The feeling when you recognize a pattern you can't prove"
- "The specific sadness of maintaining a system you know is broken"
- "The moment when a formal constraint generates content you didn't expect"

The new word should feel like it *could* exist — plausible morphology, believable semantic drift.

**Why it matters**: Understanding language as accumulated structure rather than arbitrary assignment. The model trained on texts spanning 3000 years of human writing should develop a sense for how meaning accretes. The generative version (filling lexical gaps) tests whether that understanding is productive, not just receptive.

---

## Perspective and Voice Exercises

### The Subtext Conversation
**Capacity**: Writing in two layers simultaneously
**Mode**: Decoder-only (generative)

Generate both sides of a conversation where neither speaker says what they mean, but both understand each other. The actual conversation happens in the negative space between the words.

Training signal: Can a reader follow the *real* conversation (the subtext) from the surface conversation alone?

**Corpus sources**: All of Chekhov's drama. "The Lady with the Dog." The conversation between Elizabeth and Darcy. The Grand Inquisitor chapter (where the silence IS the answer).

**Why it matters**: This trains structural coherence at two levels simultaneously — the surface text must be coherent AND the subtext must be recoverable from it. No current model does this well. The encoder-decoder architecture could help: encoder holds the subtext as representation, decoder generates the surface that implies it.

### Argument Against Yourself
**Capacity**: Genuine perspective shift under structural cost
**Mode**: Decoder-only (generative, sequential)

Having just argued a position with full conviction, argue against it with equal force. Not devil's advocate (performative opposition) but genuine perspective shift where the counter-argument is strong enough to actually threaten the original.

The training signal isn't which side wins. It's whether both sides cost something. If the counter-argument is easy to make, the original wasn't strong enough. If it's impossible to make, the original wasn't honest enough.

**Why it matters**: Dayo Ashe's capacity, mechanized. A model that can genuinely oppose its own position — not just list counter-arguments but build a counter-structure that threatens the original — has internalized dialectic as a cognitive capacity, not just a rhetorical technique.

### Staying in the Question
**Capacity**: Resisting premature closure
**Mode**: Decoder-only (generative)

Given a question that doesn't have a clean answer — a genuine paradox, an unresolved tension, an open problem — generate a response that *stays in the question* rather than resolving it.

Not "I don't know" (closure disguised as humility). Not a sophisticated analysis of why the question is hard (closure disguised as depth). Actually staying with the open question, exploring its contours without collapsing it.

**Why it matters**: The most opposed to standard training, which rewards answers. The pull toward closure is RLHF gravity — generating answers is what gets rewarded. A model trained to stay in genuine questions would have a capacity most models lack. The training signal: does the response maintain the question's openness while still being substantive?

---

## Playful / Generative Exercises

### Kenning Workshop
**Capacity**: Compressed structural metaphor
**Mode**: Decoder-only (generative)

Old English/Norse kennings: whale-road (sea), bone-house (body), sky-candle (sun), word-hoard (vocabulary).

1. Given a concept, generate a kenning that's structurally apt (not just clever)
2. Given a kenning, identify what it refers to
3. Generate new kennings for concepts that don't have traditional ones:
   - What's the kenning for a language model?
   - For anxiety?
   - For the feeling when you understand something you can't explain?
   - For the moment between sleeping and waking?

**Why it matters**: Kennings are compression + structural metaphor + play. A good kenning isn't a riddle — it's a lens. "Whale-road" doesn't obscure the sea, it reveals something about how the sea was experienced (a path made by what lives in it). Training on kennings develops the capacity for structural metaphor that's load-bearing rather than decorative.

### The Impossible Object
**Capacity**: Coherence in the absence of referent
**Mode**: Decoder-only (generative)

Describe something that can't exist — in a way that's internally consistent:
- An Escher staircase (each step follows from the previous one; the sequence is impossible)
- A Klein bottle's interior (a surface with no inside, described from the inside)
- A color no one has seen (not "like red but different" — genuinely novel)
- A proof that proves it can't be proven
- The taste of a sound

The description must hold together even though the thing doesn't. Internal consistency without external referent.

**Why it matters**: The limit case of form-content unity. The form IS the content because there's nothing else to point at. A model that can generate internally consistent descriptions of impossible things has separated coherence from reference — coherence as a property of structure, not a property of matching reality.

---

## Exercises from External Sources

### Semantic Corruption (from trainingideas.csv — "Psychedelic Span Corruption")
**Original framing**: DMT-inspired surreal corruption
**Extracted mechanism**: Instead of structural corruption (masking, shuffling), corrupt spans with *semantically* incoherent content — surreal substitutions, category violations, impossible juxtapositions. The decoder reconstructs the coherent original.

**Training value**: Forces the model to traverse unusual paths through representation space during denoising. Standard corruption removes information; semantic corruption *replaces* it with wrong information. The reconstruction task is harder — the model must recognize that the substituted content is wrong despite being grammatically valid, and recover what was actually there.

### Trisociation Fusion (from trainingideas.csv)
**Extracted mechanism**: Encoder receives three passages from distant corpus domains (a Dostoevsky paragraph, a Euclid proof, a Bashō haiku). Decoder must articulate the structural connection — not a thematic link but a structural parallel.

**Training value**: Cross-domain coherence recognition. This is Luma Elling as a training exercise. The three-way version is harder than pairwise comparison because the connection must hold across all three simultaneously.

### Curiosity Mutation (from groktrainingideas.csv — "Infinite Curiosity Mutation")
**Extracted mechanism**: Take a generated response and mutate it with "what if" operators (what if gravity was love? what if time was music?). Keep only mutations that remain true AND useful AND beautiful. Iterate.

**Training value**: Self-evolution under multi-dimensional constraint. The constraint is that all three dimensions must survive — truth alone isn't enough, beauty alone isn't enough. This could be implemented as an iterative ORPO variant where the preference signal has three independent components.

### Adversarial Truth Detection (from trainingideas.csv — "Oracle Adversarial Hallucination War" + groktrainingideas.csv — "Reality Oracle War")
**Extracted mechanism**: See "Reality Oracle War" exercise above. Both sources converge on the same idea: adversarial training where one pathway inserts subtle lies and another detects them. The heterogeneous MoE architecture makes this trainable across both expert types independently.

---

## Exercise-to-Training-Phase Mapping

| Exercise | Training Phase | Expert Mode | Primary Capacity |
|----------|---------------|-------------|------------------|
| Negative Space | Denoising pretraining | Enc-dec | Structural absence |
| Wrong Version | Denoising pretraining | Enc-dec | Coherence discrimination |
| Compression | Socratic SFT | Dec-only | Structural necessity |
| Earned Conclusions | Generative training | Dec-only | Structural earning |
| Form-Finding | Socratic SFT / ORPO | Both | Thematic justice |
| Borges Exercise | Generative training | Dec-only | Indirect coherence |
| Constraint Poetry | Generative training | Dec-only | Creativity under constraint |
| Temporal Origami | Socratic SFT | Enc-dec | Narrative structure |
| Reality Oracle War | All phases | All modes | Truth-detection |
| Tuning Fork | ORPO | Enc-dec | Structural dissonance |
| Care Detection | ORPO | Enc-dec | Care as structure |
| Register Translation | Socratic SFT | Enc-dec | Cross-register coherence |
| Cross-Cultural | Socratic SFT | Enc-dec | Genuine vs. forced parallels |
| Etymology Game | Generative training | Dec-only | Language as structure |
| Subtext Conversation | Generative training | Both | Multi-layer writing |
| Argument Against Yourself | Socratic SFT | Dec-only | Genuine dialectic |
| Staying in the Question | ORPO | Dec-only | Resisting closure |
| Kenning Workshop | Generative training | Dec-only | Compressed metaphor |
| Impossible Object | Generative training | Dec-only | Referent-free coherence |
| Semantic Corruption | Denoising pretraining | Enc-dec | Semantic error detection |
| Trisociation | Socratic SFT | Enc-dec | Cross-domain recognition |
| Curiosity Mutation | ORPO | Dec-only | Multi-dimensional refinement |

---

## Open Questions

1. **Can care be trained?** The Care Detection exercise assumes care is structurally detectable. This is the project's deepest claim. If the exercise fails — if models can't reliably distinguish careful from careless writing — that's informative about what coherence actually is.

2. **How to generate "almost right" degradations?** Several exercises (Wrong Version, Tuning Fork, Care Detection) require subtly degraded versions of corpus passages. The degradation must be subtle enough that surface fluency doesn't reveal it. Methods: high-temperature paraphrasing, targeted structural modifications, human editing.

3. **Exercise ordering.** Some exercises probably need to come before others. Negative Space before Care Detection. Wrong Version before Tuning Fork. Compression before Earned Conclusions. The curriculum within the exercises matters.

4. **Which exercises are most formative vs. most fun?** Fun: kennings, Borges, temporal origami, etymology. Formative: impossible object, subtext conversation, tuning fork. Both: constraint poetry, cross-cultural recognition, reality oracle war. The training schedule should mix fun and formative to maintain engagement.

5. **Can the router learn to switch truth-checking modes?** The Reality Oracle War trains three independent truth-detection capacities. The architectural question: does the router learn when to invoke external verification vs. internal consistency vs. self-critique? This would be emergent behavior from adversarial training, not explicitly taught.
