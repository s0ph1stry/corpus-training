# 60K Checkpoint Samples (Step 60,000 / 100,000)

**Date**: 2026-03-07
**Checkpoint**: step_060000.pt on Vast.ai RTX 8000
**Training health**: loss ~3.0-3.2, entropy 0.68-0.69, no expert death, R:70% S:14% X:15%

## R-Mode (Denoising with Encoder, ~30% masked)

```
ORIGINAL: The sea was calm and the lighthouse stood firm against the morning fog.
OUTPUT:   . . . calm and the day gh . . stood up the the morning . .
(9/19 tokens masked)

ORIGINAL: In the beginning was the Word, and the Word was with God.
OUTPUT:   In the beginning of the Word of the the Word was with God .
(4/16 tokens masked)

ORIGINAL: The fundamental problem of political economy is not scarcity but distribution.
OUTPUT:   The fundamental problem of political the . of the the the the
(7/14 tokens masked)

ORIGINAL: She opened the door and found nothing but silence and a single letter on the floor.
OUTPUT:   She had a door and found the but she and a single letter on the floor of the
(6/19 tokens masked)

ORIGINAL: Energy return on investment determines the complexity a civilization can sustain.
OUTPUT:   . . gy return on invest ment determines the other of a few can sustain . The
(6/18 tokens masked)

ORIGINAL: We hold these truths to be self-evident, that all men are created equal.
OUTPUT:   accidentally We , , , to be self , evident , that all men are created equal . .
(6/19 tokens masked)
```

**Assessment**: Partial reconstruction. Gets structure and common words right. Struggles with domain-specific vocabulary and heavily masked regions. The John 1:1 reconstruction is closest — "In the beginning of the Word" vs "was the Word" (KJV likely well-represented in corpus).

## S-Mode (Autoregressive, Greedy)

```
PROMPT: The lighthouse keeper watched
OUTPUT: ...and the world of the world, and the world of the world...

PROMPT: In the matter of political economy
OUTPUT: ...and the same, and the same, and the same...
```

**Assessment**: Classic early-training repetition. Model has learned common n-grams but no long-range coherence. Expected at 60%.

## S-Mode (Autoregressive, temperature=0.8, top-k=50)

```
PROMPT: The lighthouse keeper watched
OUTPUT: The lighthouse keeper watched te, and he said to the earth, 'My head are
in the world.' "'But a man,' said the Hain; 'and if there's not a man's
know, of which he may be with man,' and do not tell him.''

PROMPT: In the matter of political economy
OUTPUT: In the matter of political economyence, and the most of the most equal,
the same, or of the the same, and the _ the _ The _ A _ is...

PROMPT: The energy of civilization
OUTPUT: The energy of civilizationdity, and on the cause of the old man, and to
have much of the other, in which one would be the world, but the most
important and good is to be made in the first, and and of all of us we
must not find, as we have not that the question

PROMPT: She walked into the room
OUTPUT: She walked into the roomles. She had known her things, and had a good of
her own mother, even he would have been in her a right of his own and but
as a day he began to the house, and the woman asked. "He was for the great
name of my own?" said Diam

PROMPT: Once upon a time there was
OUTPUT: Once upon a time there waske. And with no longer I had no good own; I had
been at his work. I could not have been but one — I would be so much as I
could be at my good, and I might not know good; and in my good work of us
was seen to have a man

PROMPT: God said unto
OUTPUT: God said unto. For the true is is in that of God, and this is not the
truth's; but yet therefore he cannot be good and not to be but the Lord as,
and that which is to do and, and must not be the "in the present," for I
have not,
```

**Assessment**: Temperature breaks repetition. Model shows:
- Register awareness: theological for "God said", narrative for "She walked", first-person for fairy tales
- Dialogue structure: quotation marks, "said X", conversational turns
- Pronoun tracking within short spans (she/her, he/his)
- NOT YET: long-range coherence, factual content, clean subword boundaries

**Prognosis**: Remaining 40K steps with increasing difficulty curriculum should significantly improve coherence. The register awareness and structural learning are encouraging foundations.
