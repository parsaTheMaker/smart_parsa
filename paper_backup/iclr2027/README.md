# ICLR 2027 Submission Workspace

This directory is the working home for the ICLR 2027 submission on sampling-density-invariant neural surrogates.

## Layout

- `official/`: downloaded ICLR materials and offline copies of the official policies. Do not edit these files.
- `template/`: untouched extraction of the official ICLR 2027 LaTeX starter kit.
- `manuscript/`: working anonymous LaTeX copy. Start from `manuscript/iclr2027_conference.tex`.
- `figures/`, `tables/`, `appendix/`: submission-ready assets, source tables, and appendix-only material.
- `artifacts/`: anonymous code/data/supplementary packages prepared for OpenReview.
- `notes/`: project framing and writing plan.
- `submission/`: final preflight checklist and submission records.

## Project Framing

The current project studies a failure mode of neural surrogates on unstructured three-dimensional geometry: predictions can change when the same continuous surface is represented by a point cloud with a different spatial sampling density. The proposed SATLOSS training objective pairs two density-shifted representations of the same geometry, supervises both against the same physical target, and explicitly aligns their predictions.

The latest presentation with Nils Thuerey is at `../presentations/2026-08-19/meeting_20260819.pdf`. The key paper claim must be supported by: (1) a precise invariance target, (2) broad architecture coverage, (3) controlled point-distribution shifts and remeshed geometry inputs, (4) accuracy as well as robustness, (5) ablations isolating density estimation, shift range, and the consistency term, and (6) reproducible code and data processing.

## Start Writing

1. Keep `\\iclrfinalcopy` commented out and retain anonymous authorship in the submission PDF.
2. Write the abstract and introduction around the problem, not around a single dataset: the object of invariance is the representation of a fixed geometry.
3. Reserve main-paper space early for the controlled protocol, strongest comparisons, and ablations. Put exhaustive plots, per-field metrics, implementation details, and proofs in the appendix.
4. Use only source-controlled vector figures and generated tables. Every headline number should be reproducible from a saved result artifact.
5. Track all generative-AI use from now on. ICLR 2027 requires a paper-level AI-use statement and a matching OpenReview disclosure.

Read [the project plan](notes/PROJECT_AND_WRITING_PLAN.md) and [the preflight checklist](submission/ICLR2027_PREFLIGHT_CHECKLIST.md) before drafting.

## Official Sources

- Author guidelines: <https://iclr.cc/Conferences/2027/AuthorGuidelines>
- Call for papers: <https://www.iclr.cc/Conferences/2027/CallForPapers>
- AI policy for authors: <https://iclr.cc/Conferences/2027/AIPolicyForAuthors>
- Submission portal: <https://openreview.net/group?id=ICLR.cc/2027/Conference>
- Official template archive: <https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>

Downloaded on 2026-08-24. Always recheck the official pages before final submission.
