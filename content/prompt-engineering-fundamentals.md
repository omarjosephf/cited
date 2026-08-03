# Prompt Engineering Fundamentals

A practical guide to writing instructions for large language models. Written for
professionals who use AI tools in their work but do not build them.

## What a prompt actually is

A prompt is the complete input a language model receives before it generates a
response. It is not only the question typed into a chat box: it also includes any
system instructions set by the application, the conversation history, and any
documents or data supplied alongside the question.

This matters because models have no memory between separate conversations. Every
request is answered from the prompt alone. If information is not in the prompt,
the model cannot use it, no matter how many times it has been told before.

## The four components of a well-formed prompt

Most effective prompts contain four parts. They can appear in any order, but
leaving one out is the most common cause of a disappointing result.

1. **Role** — who the model should act as. "You are a financial analyst
   reviewing a quarterly report."
2. **Task** — what it should produce. "Summarise the three largest changes in
   operating costs."
3. **Context** — the material it should work from, and any constraints.
   "Use only the figures in the attached statement."
4. **Format** — the shape of the output. "Return a bulleted list, one line per
   change, with the percentage in brackets."

A prompt missing the format component is the most frequent source of output that
is correct but unusable.

## Specificity beats politeness

Models do not respond to enthusiasm. "Please give me a really great summary" adds
tokens without adding information. "Summarise in under 100 words, for a reader
with no accounting background" changes the output measurably.

Replace subjective adjectives with checkable constraints:

- "Make it short" becomes "no more than 150 words"
- "Make it professional" becomes "no contractions, no exclamation marks"
- "Explain it simply" becomes "assume no prior technical knowledge"

The test is whether a second person could look at the output and agree it met the
instruction. If they could not, the instruction was not specific enough.

## Show the model what good looks like

Supplying worked examples inside the prompt is called few-shot prompting.
Providing no examples is called zero-shot prompting.

Few-shot prompting is most valuable when the required output has a shape that is
hard to describe but easy to demonstrate — a particular tone, an unusual
structure, a house style. Two or three examples are usually enough. Beyond about
five, the added benefit falls away sharply while the cost of every request keeps
rising.

Examples should show the desired outcome rather than the mistake to avoid.
Demonstrating what you want works more reliably than describing what you do not.

## Give the model room to reason

For any task involving several steps — multi-stage calculations, comparisons
against criteria, decisions with conditions — asking for the answer alone tends
to produce worse results than asking for the reasoning first.

Instructing the model to work through the problem before stating a conclusion is
known as chain-of-thought prompting. The improvement is largest on arithmetic and
logical tasks and smallest on straightforward retrieval, where the reasoning adds
cost without adding accuracy.

Newer reasoning models perform this step internally and do not need to be asked.
Instructing them to "think step by step" can make responses longer without making
them better.

## Constrain the source, not just the answer

The most useful control in professional work is telling the model what it is
allowed to draw on.

An instruction such as *"answer using only the document provided; if the document
does not contain the answer, say so"* changes model behaviour substantially. It
converts a general-knowledge system into one grounded in a specific source, and
it makes a wrong answer far easier to detect, because a claim not present in the
source is visibly out of scope.

This is the principle behind retrieval-based systems: the model is given relevant
passages and instructed to answer from them alone. Grounding does not eliminate
error, but it makes error checkable, which is the property that matters in
professional use.

## Iterate deliberately

Treat prompts as drafts. The reliable improvement loop is:

1. Write the prompt and run it.
2. Identify what is specifically wrong with the output — not "it is bad" but "it
   invented a statistic" or "it used the wrong tone".
3. Change one thing.
4. Run it again on the same input.

Changing several things at once makes it impossible to know which change helped.
Keeping the input fixed between runs is what makes the comparison meaningful.

## Where prompting stops being enough

Prompting has limits, and recognising them prevents wasted effort.

- **Missing information.** No prompt can make a model report facts it was never
  given. The fix is supplying the information, not rewording the request.
- **Currency.** A model cannot know events after its training cutoff. The fix is
  retrieval or a search tool, not a better prompt.
- **Reliability at scale.** A prompt that works nine times in ten is unsuitable
  for automation. The fix is validation, structured output constraints, or human
  review — not more instructions.
- **Consistency of style across an organisation.** Beyond a certain volume, this
  belongs in tooling and templates rather than in each individual's prompt.

## A short checklist

Before sending a prompt for anything that matters:

- Is the role stated, if it affects the answer?
- Is the task a single, clear instruction?
- Is the source material supplied, and is the model told to stay within it?
- Is the output format specified precisely enough to be checked?
- Are the constraints measurable rather than subjective?
- If the task has multiple steps, has the model been asked to reason first?
- Is there a defined way to tell whether the output was acceptable?

A prompt that passes this checklist will not always produce a good answer. It
will, however, produce an answer whose quality can be judged — which is the
prerequisite for improving anything.
