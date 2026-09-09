# Paste this into the writing chat

**Attach four files:**

1. `notes/paper/TMS_paper_v12_draft.docx` — the draft to edit
2. `notes/paper/TMS_paper_v12_draft.txt` — same text, flat; the audit's ¶ numbers are lines in THIS file
3. `notes/FOR_THE_WRITING_CHAT.md` — the audit and the settled arguments
4. `notes/PLACEHOLDERS.md` — the token table

---

## Prompt

I'm finishing v12 of a paper (combined cTBS-TMS + 7T fMRI study of parietal
magnitude representations and risk attitudes). The draft is attached along with
an audit of it produced from the analysis repo. Please revise the draft against
that audit.

**Read `FOR_THE_WRITING_CHAT.md` first.** Its last major section, "Audit of
TMS_paper_v12_draft.docx", is keyed to paragraph numbers in the attached `.txt`.
Sections 1–11 above it carry the settled arguments and the wording I want used.

Work in this order:

1. **§A — the one actual error.** ¶80 claims the model "converged without any
   adjustment to sampler or priors (r̂ ≤ 1.002, ESS ≥ 1727)". Those numbers are
   from a fit under the old choice rule; under the rule the paper describes, the
   same model gives r̂ 1.12 / ESS 42. Rewrite that sentence with ⟦RHAT⟧ / ⟦ESS⟧
   and drop the "without any adjustment" claim.

2. **§B — tokenise.** Replace each listed number with its ⟦TOKEN⟧, exactly as
   spelled in `PLACEHOLDERS.md`. Keep the brackets intact and don't let one
   break across a line — they're substituted mechanically later. Every token has
   a current best estimate in that file, so judge each sentence for sense with
   the estimate in mind, then write the token.

3. **§C — the Methods gaps.** These are missing sentences, not wrong ones.
   § Model estimation has no sampler settings, no convergence criterion, no
   prior specification and no statement of how ELPD was computed — add all four.
   § The flexible PMC model documents only the spline model, but Results reports
   the power law; the reported model needs a Methods entry. And add one sentence
   stating that every model in the paper, including the probit, is hierarchical
   Bayesian with partial pooling, and that no maximum-likelihood estimate or
   bootstrap CI appears anywhere.

4. **§D — prose.** Four changes, each with the reasoning given in the audit. The
   important one is ¶80's justification for the reported model: as written it
   implies model comparison settled which channel carries the effect, and it
   didn't. Say what ELPD does establish, say plainly that it does not adjudicate
   between placements, then give the reason we chose as we did.

5. **§E — leave alone.** ¶75, ¶86, ¶82's framing, the Figure 5 caption, the
   Weber section and the participants paragraphs are right. Don't polish them.

Constraints:

- **Don't invent a number.** If a value isn't in `PLACEHOLDERS.md` and isn't
  already in the draft, write ⟦TODO: what it is⟧ and list it at the end.
- ⟦BB_INTERVAL⟧ in ¶85 is genuinely unknown — the fit that produces it hasn't
  converged. Leave the brackets and write no number.
- Match the draft's register. It's a Nature-Comms-style paper; don't make it
  breezier or add signposting it doesn't already use.
- Return the revised text section by section with a short note on what changed,
  not a diff.

Then give me a list of every ⟦TOKEN⟧ that ended up in the manuscript, so I can
check it against the substitution table.
