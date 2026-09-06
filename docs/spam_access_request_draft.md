# Draft — SPaM'17 CTG Challenge data access request

**Status: DRAFT, NOT SENT.** Outward-facing correspondence is the project owner's
to send. Verify the recipient before use.

**Recipient:** Dr Antoniya Georgieva (organiser, SPaM in Labour Workshop / CTG
Challenge 2017), Nuffield Department of Women's & Reproductive Health, University
of Oxford. The challenge page
(<https://users.ox.ac.uk/~ndog0178/CTGchallenge2017.htm>) states the dataset
"has now been removed" and directs interested parties to contact her.

**Purpose disclosure — read before sending.** The page invites contact for the
dataset's *original purpose — to validate existing algorithms*. Our intended use
is **self-supervised pre-training**, which is broader than that. The draft below
states this plainly rather than framing our work as validation. If the answer is
no, that is a clean scientific result for the project and is recorded as such.

---

**Subject:** Request for access to the SPaM'17 intrapartum CTG dataset — MSc
research on CTG-based acidaemia prediction

Dear Dr Georgieva,

I am writing to ask whether access to the SPaM in Labour 2017 CTG Challenge
dataset is still possible. The challenge page indicates the data has been removed
from public download and directs enquiries to you.

**Who I am.** I am conducting research on intrapartum CTG-based prediction of
fetal acidaemia, using the open CTU-UHB database as the benchmark.

**What I would use it for, stated plainly.** My intended use is *not* the
challenge's original purpose of validating an existing algorithm. It is
**self-supervised pre-training** — masked-signal reconstruction on unlabelled
CTG, followed by fine-tuning and evaluation entirely on CTU-UHB. I understand
this is a broader use than the page describes, which is why I am asking rather
than assuming.

**Why this dataset specifically.** Recent work (Fridman & Ben Shachar, 2026,
arXiv:2601.06149) reports that pre-training on the CTGDL corpus — CTU-UHB plus
FHRMA plus SPaM'17, 984 recordings and 2,444 hours — substantially improves
CTU-UHB performance. I have reproduced the openly available part of that corpus
via CTGDL v5 on Zenodo, but SPaM'17 accounts for roughly two thirds of those
hours and is the one component that cannot be redistributed. Without it I can
assemble only about 34% of the corpus, which my own analysis suggests is
unlikely to be sufficient.

**What I would commit to.**

- Signing whatever Data Use Agreement you require, including institutional
  counter-signature.
- Using the data solely for pre-training; **no SPaM recording would appear in any
  evaluation set**, and no outcome labels from SPaM would be used.
- No redistribution of the data in any form, including derived signal files or
  model checkpoints from which signals could be reconstructed.
- Citing the dataset and the SPaM in Labour workshop in any resulting work.
- Deleting the data on request or at the end of the project.

**A smaller alternative, if full access is not appropriate.** If pre-training use
falls outside what the DUA permits, I would equally welcome the narrower
arrangement the page describes — evaluating my existing model on the dataset as
an external validation cohort, with results reported back to you. That would be
scientifically valuable to me and is closer to the stated original purpose.

I am happy to provide a fuller research protocol, institutional details, or an
ethics reference on request.

With thanks for your time,

[Name]
[Institution / supervisor]
[Email]

---

## Notes for the sender

- Attach or link the project's evaluation protocol
  ([`src/training/protocol.py`](../src/training/protocol.py)) if a protocol is
  requested — it demonstrates patient-grouped evaluation and no test-fold
  selection, which supports the "no leakage" commitment above.
- Do **not** overstate institutional backing; if a supervisor must
  counter-sign a DUA, say so upfront.
- Record the outcome (granted / refused / no reply by a set date) in
  [phase9_data_availability.md](phase9_data_availability.md) either way. A
  refusal bounds the project's external-data ceiling and is a reportable result,
  not a dead end.
