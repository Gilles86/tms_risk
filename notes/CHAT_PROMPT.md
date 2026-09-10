# Paste this into the writing chat

**Attach four files:**

1. `notes/paper/TMS_paper_v12_draft.docx` — the draft to edit
2. `notes/paper/TMS_paper_v12_draft.txt` — same text, flat; the audit's ¶ numbers are lines in THIS file
3. `notes/FOR_THE_WRITING_CHAT.md` — the arguments, the wording, and the audit of v12
4. `notes/PLACEHOLDERS.md` — the token table

---

## Prompt

I'm finishing v12 of a paper (combined cTBS-TMS + 7T fMRI study of parietal
magnitude representations and risk attitudes). The draft is attached with an
audit of it produced from the analysis repo. Please revise the draft against
that audit.

**Read `FOR_THE_WRITING_CHAT.md` first.** Sections 1–11 carry the settled
arguments and the wording I want used; the section "Audit of
TMS_paper_v12_draft.docx" near the end is keyed to paragraph numbers in the
attached `.txt`.

The model analysis changed substantially since the draft was written, so start
there:

1. **§2 — the reported model changed.** The draft reports a model in which cTBS
   acts on the first- and second-presented options separately. It now reports
   ⟦MODEL_NAME⟧: the STAGE-indexed parameterisation (perceptual + memory noise)
   with the cTBS effect on perceptual noise and on the two magnitude-prior
   means. §2 gives the reasoning and, importantly, the reasoning NOT to use —
   do not write that the position-indexed model cannot be fitted.

2. **§9 — prior shifts.** Earlier drafts said prior shifts are excluded. Prior
   WIDTHS still are, on a stated principle; prior MEANS are now part of the
   reported model. §9 has the wording.

3. **§3 — the numbers.** Replace each with its ⟦TOKEN⟧ exactly as spelled in
   `PLACEHOLDERS.md`. Keep the brackets intact and never split one across a
   line. Every token has a current best estimate there, so judge each sentence
   for sense with that value in mind, then write the token.

4. **The withdrawn claims.** Two things in earlier material are retracted and
   must not appear: any claim that the noise increase is specific to the
   second-presented option, and the figure that the model recovers ~40% of the
   order asymmetry. Both came from a non-converged fit. See the
   "order asymmetry — WITHDRAWN" section.

5. **§B/§C/§D of the audit.** Remaining numbers to tokenise, three whole
   Methods gaps (no sampler settings, no convergence criterion, no priors, no
   LOO method; and the reported model has no Methods entry), and four prose
   changes.

6. **§E — leave alone.** Those paragraphs are right; don't polish them.

Constraints:

- **Don't invent a number.** If a value isn't in `PLACEHOLDERS.md` and isn't
  already in the draft, write ⟦TODO: what it is⟧ and list it at the end.
- ⟦BB_INTERVAL⟧ is genuinely unknown — the fit that produces it has not
  converged. Leave the brackets and write no number.
- Model comparison establishes three things and not a fourth: see §3's ladder
  table. Do not let the prior-mean shift be presented as established by ELPD;
  the posterior predictive checks carry it.
- Match the draft's register — a Nature-Comms-style paper. Don't add
  signposting it doesn't already use.
- Return the revised text section by section with a short note on what changed,
  not a diff.

Then list every ⟦TOKEN⟧ that ended up in the manuscript, so I can check it
against the substitution table.
