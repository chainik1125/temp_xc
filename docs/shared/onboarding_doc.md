---
title: "Onboaring document"
author: Dmitry Manning-coe
date: 2026-02-16
type: document
---

# Welcome to Temporal Crosscoders!

Author: Dmitry Manning-Coe
Date: 2026-02-16


## Logistics

Please do the following:

1. Fill out the Timeful for the weekly meeting (https://timeful.app/e/8AcAb). These will start at 1 hour and we will adjust when need be.
2. Fill out the Timeful/When2meet for weekly 2-1s. These will start at 30m and we will adjust as need be.
3. Please create a folder in temp_xc/docs/YourFirstName that contains your time zone, expected working hours/days etc...
4. Please set up a 30m call with your team mate! I'm hoping that pairing people together will give you an extremely valuable close-collaborator for low-level work/ideation that can often be missing in projects (More on this in the [[manifesto|manifesto document]]...)

## Getting started with the repo

### Writing things up

All important pieces of theory, experimental design, results, and discussion should be written up as markdown notes. You can write in HackMD (for collaboration) or directly in `docs/` using VSCode -- both work. When your note is ready, make sure it lives in `docs/` with the required frontmatter (author, date, tags) and run `./run-checks.sh` before pushing.

### References

When you cite a paper, add an entry to [[references]] with the citation key, full reference, link, and a brief line about why it's relevant. Then link to it from your doc with `[[references#citationkey]]`. Keeping references centralized means everyone benefits when someone adds a new paper.

### Tooling

- **Python**: run `uv sync` to set up the environment
- **Claude Code**: this repo has built-in AI review commands (`/review`, `/review-doc`, `/review-math`, etc.) that you can run from Claude Code. If you don't have access to Claude Code, let Dmitry know.
- **VSCode extensions**: when you open the repo in VSCode you'll be prompted to install the recommended extensions (HackMD, Markdown Preview Enhanced). These are optional but make writing docs nicer.

## Literature

All citations live in [[references]]. If you have time I would recommend beginning to go through them in the order here.

Key reading:

1. For our initial experiments on synthetic data we will be using the setup I talked about in the interview: [[references#manningcoe2025synthetic]]

2. This is based on [[references#lindsey2025synthetic]] (TODO: Add David's new synthetic benchmark paper)

3. Important foundational literature for thinking about how features are represented:
    - [[references#elhage2022toy]]
    - [[references#bricken2023monosemantic]]
    - [[references#templeton2024crosscoders]]

4. Important recent work on Temporal XC:
    - [[references#dunefsky2025txc]]
    - [[references#kissane2025txc]]

5. Empirical work on why temporal features might be useful
    - [[references#venhoff2025reasoning]]
    - [[references#ward2025reasoning]]

