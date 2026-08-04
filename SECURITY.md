# Security

## Reporting a vulnerability

Email **ojflorendo.connect@gmail.com** with "SECURITY" in the subject.

Please include what you found, how to reproduce it, and what you think the
impact is. I will acknowledge within **72 hours** and tell you what I intend to
do about it, including if the answer is "nothing, and here is why".

Please do not open a public issue for anything exploitable. Please do not run
automated scanners or load tests against a deployed instance — see *Testing*
below.

This is a personal project maintained by one person. I cannot promise the
response times an organisation could, and saying so is more useful than a policy
I would not meet.

## What this software is

A document question-answering service. It retrieves passages from a fixed corpus
of documents, sends them to a language model with the user's question, and
returns an answer with citations back to the source passages.

Two properties matter for reasoning about its security:

- **The corpus is authored, not uploaded.** Documents are committed to this
  repository. There is no upload path, so there is no route for an attacker to
  place content into the corpus without a pull request being merged.
- **The user controls only the question.** Questions are treated as data, never
  as instructions, and this is stated explicitly in the system prompt.

## Known limitations

Stated plainly, because a security policy that lists only strengths is
marketing.

**Prompt injection is mitigated, not solved.** The system prompt instructs the
model to treat both documents and questions as data and to ignore anything in
them that resembles a command. That reduces the risk; it does not eliminate it.
No current technique does. The practical consequence is bounded: the model's
only capability is producing text from passages already supplied, so a successful
injection can produce misleading output but cannot read other users' data, reach
another system, or execute anything.

**Citations are verified; answers are not.** Every citation is checked against
the passage actually supplied, and a quote that does not appear in it is
discarded (`src/assistant/answering.py`). That guarantees a citation points at
real supplied text. It does not guarantee the surrounding claim is a correct
reading of that text. Verifiability is the property this project provides;
correctness still requires a reader.

**Refusal is a judgement, not a guarantee.** The model decides whether the
passages answer the question (see `docs/adr/0002-*`). It can be wrong, in both
directions. `doc-assistant eval` measures how often.

**No authentication.** A deployed demo is public by design. Anything sent to it
should be treated as public.

**Third-party processing.** Questions and retrieved passages are sent to the
Anthropic API. Their handling is governed by Anthropic's terms, not by this
project.

## If you deploy this

Two controls are not optional, and neither is enforced by this code:

1. **A hard spend cap** on the API account. An unauthenticated endpoint that
   makes paid calls is a financial denial-of-service waiting to happen. Rate
   limiting narrows the window; only a spend cap bounds the loss.
2. **Rate limiting**, per IP, at the edge as well as in the application.
   Application-level limits do not survive a process restart and do nothing
   about traffic that never reaches the process.

Do not put confidential documents in the corpus of a public deployment. Every
passage in it is retrievable by asking the right question — that is what the
software is for.

## Secrets

`ANTHROPIC_API_KEY` is the only credential. It is read from the environment or
a `.env` file, which is git-ignored, and is held as a `SecretStr` so it renders
as `**********` in logs and tracebacks rather than in plain text.

`tests/test_no_secrets_committed.py` fails the build if any credential reaches a
tracked file, and runs as its own CI job. It exists because a key was once
pasted into `.env.example` — the tracked template — instead of `.env`. It was
caught before being committed, and the guard is there so the next one does not
depend on being caught by eye.

**A credential written into a tracked file should be treated as compromised even
if it was never committed.** Rotate it rather than reasoning about whether anyone
saw it.

## Testing

Against your own local instance: anything you like.

Against a deployed instance: please do not run automated scanning, fuzzing or
load testing. Every request costs the operator money, so a scan is
indistinguishable from an attack on their bill. Email first and I will point you
at a local setup instead.

## Supported versions

The `master` branch is the only supported version. This is a project in active
development, not a release-managed product; fixes land there and nowhere else.
