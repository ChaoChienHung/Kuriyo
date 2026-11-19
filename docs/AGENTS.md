# Kuriyo docs/ wiki contract

> This directory is a **wiki maintained by an LLM**: linked markdown pages, not a pile of
> flat documents and not a separate site.
> This file defines the rules. **Change the structure only with the user's approval.**

This is the **single source** of the wiki rules for every agent (Claude, Codex, or any
other). The repository root `AGENTS.md` says *how to work*; this file says *how to keep
the documentation*. Where they overlap, link -- do not copy the text across.

`docs/CLAUDE.md` only imports this file. It exists because Claude Code auto-loads files
with that name and the `llm-project-docs` plugin checks for it; keep it a one-line import
and never put rules there. The `/docs-*` slash commands in `.claude/commands/` are
likewise one-line pointers to [Procedures](#procedures) below. Any agent can follow the
procedures directly, so no tool-specific copy is needed: do not commit auto-converted
copies such as `.agents/skills/source-command-docs-*`.

## Why this shape

A single large document loses three things.

1. **Progressive disclosure** -- answering one question means reading all of it.
   `index.md` (small) then 2-3 relevant pages is far cheaper
2. **A sync mechanism** -- no way to compute which document drifted from the code
3. **A single source of truth** -- the same fact in several documents will diverge

So every page carries `verified_at` (the commit it was checked against), and an update
reads **only what changed since that commit** instead of re-reading everything.

## Language

Pages are written in **Traditional Chinese**. Frontmatter keys, `[[link]]` targets, file paths and the
`/docs-*` commands stay as they are, whatever the language of the prose. Page file names
may be in Traditional Chinese; the linter and hooks handle non-ASCII names. To change the language,
change this line -- every agent working on this repository reads it before writing.

## Pages are categorised by access pattern

| | Categories |
|---|---|
| **Exploratory** | `concepts` `architecture` `subsystems` `decisions` |
| **Lookup** | `reference` `operations` |
| **Historical** | `history` |

**Exploratory** pages are read when you do not know where to look, so `index.md` and
`[[links]]` must be dense. **Lookup** pages are opened when you already know -- tables,
kept short. Decide which kind a new page is before writing it.

## Layout

```
docs/
├── AGENTS.md      this file: the wiki rules and procedures
├── CLAUDE.md      one-line import of AGENTS.md for Claude Code
├── index.md       table of contents + reading order
├── log.md         work log (append-only)
├── raw/           source material that exists neither in code nor in pages
│
├── concepts/       Domain and product concepts. Mostly survive code changes
├── architecture/   System structure, data model, main flows
├── subsystems/     Per-subsystem implementation
├── decisions/      Why it was decided that way (ADRs, known defects)
├── reference/      Schemas, APIs, env vars -- documents you look values up in
├── operations/     Install, deploy, checks, incident response
├── history/        Development history, lessons, migrated records
```

## Page format

```yaml
---
title: Page title
type: concept | architecture | subsystem | decision | reference | operation | history
created: YYYY-MM-DD
updated: YYYY-MM-DD          # the day the body actually changed
sources:
  - code: src/services/search.py      # path from the repository root
  - raw:  YYYY-MM-DD_meeting-notes.md
  - ext:  https://...
verified_at: <short sha>              # the commit this page was checked against
---
```

Rules:

- Mentioning another page means linking it: `[[page-name]]` (file name, no path)
- No claim without a source. If there is no evidence, write `(unverified)`
- Cite code as `path/to/file:line`. **Do not paste code in.**
  Describe what it does and why, and point at where it lives
- **Write every trap as its own `> ⚠️` paragraph.** A trap is something that bit, or will
  bite, whoever changes this code: an ordering that matters, a setting that must stay, a
  fix that looks wrong but is right. Before a file is edited, the plugin's hook shows the
  `⚠️` paragraphs of the pages that cover it, and nothing else. A trap written as plain
  prose is never shown at the moment it matters
- `updated` is the day the body changed. If you only confirmed the sha and the prose
  still holds, move `verified_at` alone
  - Note that `/docs-lint` reports such a page as **unverified**
    ("sha moved past N code commits with no body change"). That is not a prohibition --
    it marks a *weak* verification. `verified_at` is a declaration, not proof, and this
    is the one place where "did you actually read it?" becomes visible
- **One fact lives on one page.** Link from anywhere else that needs it.
  In particular, a lookup page and an exploratory page must not each carry the same table

## What counts as code

This is how `/docs-sync` and `/docs-status` decide "did the code change".

```
code = the whole repository - { docs , .claude , CLAUDE.md }
```

As a git pathspec, used verbatim by the procedures below (if you change the definition,
change every copy in [Procedures](#procedures) in the same edit):

```bash
git log <range> -- . ':(exclude)docs' ':(exclude).claude' ':(exclude)CLAUDE.md'
```

It is an **exclude list**, not an include list. When a new code directory appears an
include list silently misses it; an exclude list picks it up automatically.

> ⚠️ The root `AGENTS.md` **is code** for this purpose: it holds repository conventions,
> and pages such as [[model-family-layout]] cite it as a source. Do not "fix" the
> exclude list by swapping `CLAUDE.md` for `AGENTS.md` -- that is exactly what automatic
> conversion tools do, and it hides convention changes from `/docs-sync`.

> Because the docs live in the same repository as the code, **editing only docs still
> moves HEAD.** Without this filter `/docs-status` keeps reporting "N commits behind"
> when there is nothing to do, and then nobody believes it any more.
>
> This is **a filter for judgement, not a version-control exclusion.** `docs/` and
> `.claude/` are committed alongside the code on purpose -- documentation has to travel
> with the repository for whoever clones it to keep it up.

## Workflow

Each name is a procedure in [Procedures](#procedures). Claude Code also exposes them as
slash commands; other agents run the same steps directly.

- `/docs-sync` -- read only what changed and update affected pages. **Stop immediately if
  no code changed**
- `/docs-lint` -- orphans, broken links, broken sources, stale, unverified
- `/docs-query` -- answer from the docs only. If it is not there, say so
- `/docs-status` -- commits not yet reflected, stale pages, recent work

## When you change code

| What changed | Page to update |
|---|---|
| 模型家族或變體目錄、runs 位置、`scripts/run.sbatch` 的 run 路徑 | [[model-family-layout]] |
| `Representations/SID/` 的 RQ-VAE、optimizer、碼本初始化或診斷工具 | [[sid-rqvae-training]] |
| `Recall/FMRec/` 的模型、訓練、評估或預設值 | [[fmrec]] |
| `Recall/SASRec/` 的模型、訓練資料、評估或預設值 | [[sasrec]] |
| `Recall/TIGER/baseline/train.py` 的 T5 設定、optimizer 或學習率 | [[tiger-training]] |

**Rhythm**: run `/docs-sync` right after a deploy or a release -- the moments when
changes pile up. Do not wait to be asked.

**Releases**: a deploy or a release moves no commit, so the plugin cannot see one unless
the line below names it. Write the commands that deploy or release this project as one
regular expression (for example `fastlane|docker compose .* up|npm publish`). When an
agent runs a matching command, the commit hook says `/docs-sync` is due. Leave it empty if
there are none.

release-commands:

## Security

**Never record secret values.** Variable names at most.

Repo 會記錄 NUS SoC Slurm 的 partition、job ID 與叢集路徑，但不記錄帳號密碼、token 或 Hugging Face 憑證。

## This wiki grows as you work

docs-mode: grow

It was started small: a skeleton and a page or two. It is filled in where work happens,
not all at once.

- **When you finish work in an area no page covers, add a page for it, or extend the
  nearest page.** What the code does, why it is that way, and every trap you hit as a
  `> ⚠️` paragraph, with `sources` pointing at the files you worked in. Keep it short;
  it will grow. The commit hook names files a commit touched that no page covers yet.
- **Do not document areas you did not work in.** That is what `/project-docs-setup`
  does later, when someone wants the whole picture: it fills the uncovered areas and
  leaves the pages written here alone.

## Procedures

Run every command from the repository root.

### `/docs-sync` -- reflect code changes into the docs wiki

#### Procedure

> ⚠️ One syncer at a time, and pull first. Two sessions syncing at once both append
> `docs/log.md` and edit `docs/index.md`, and the second one conflicts. In a multi-session
> setup, let one session sync and the others wait.

##### 1. Baseline

The **oldest** `verified_at` sha across `docs/**/*.md` frontmatter.

##### 2. First check whether any code changed

**Do not count docs commits.** The docs live in the same repository, so editing only
docs still moves HEAD. Without this filter you keep seeing "N commits behind" when
there is nothing to do.

```bash
git log <baseline>..HEAD --oneline -- . ':(exclude)docs' ':(exclude).claude' ':(exclude)CLAUDE.md'
```

**If it is empty, stop: "no code changes, /docs-sync not needed".**
Leave `verified_at` alone -- the code has not moved, so it is still valid.

##### 2-1. Collect changed files

```bash
git log <baseline>..HEAD --name-only --pretty=format: -- . ':(exclude)docs' ':(exclude).claude' ':(exclude)CLAUDE.md' \
  | sort -u | grep -v '^$'
```

> Git quotes non-ASCII paths by default (`core.quotepath`), so a path with
> non-Latin characters comes back escaped and a naive `grep '^docs/'` misses it.
> Use `-z`, or `git -c core.quotepath=false`, when a filename might not be ASCII.

##### 3. Select affected pages

Match changed file paths against each page's `sources[].code` by **prefix**.
Report the selection first: N changed files -> M affected pages.

##### 4. Update

For each page, **read the actual diff, not the commit message**
(`git diff <baseline>..HEAD -- <path>`).

- Prose still correct -> move `verified_at` only
- Prose wrong -> fix the body, and move both `updated` and `verified_at`
- A cited `file:line` that shifted -> new line number

> ⚠️ `verified_at` is written *before* you commit, so it names the commit before yours.
> A docs-only commit does not make anything stale: staleness counts only commits that
> touch a page's `code:` sources. If one commit changes both code and the page that
> describes it, that page shows "stale by 1" -- commit code and docs separately.

##### 5. Unclassified changes

List files that matched no page and **propose new pages**. Do not create them without
approval.

##### 6. Finish

- Update `docs/index.md` (if pages were added)
- Append to `docs/log.md`
- Verify with `/docs-lint`

#### Principles

- **Only read what changed.** Do not re-read the whole codebase every time
- State lives in `verified_at` and nowhere else. Do not add a status file

### `/docs-status` -- docs wiki status

#### 1. Commits not yet reflected (first)

Against the oldest `verified_at` across all pages. **Count code and docs separately** --
a docs-only commit is not something `/docs-sync` acts on.

```bash
git log <baseline>..HEAD --oneline -- . ':(exclude)docs' ':(exclude).claude' ':(exclude)CLAUDE.md' | wc -l   # code
git log <baseline>..HEAD --oneline -- docs .claude | wc -l   # docs
```

```
<baseline>..HEAD   code 0 commits - docs 1 commit    -> /docs-sync not needed
<baseline>..HEAD   code 4 commits - docs 0 commits   -> /docs-sync recommended
```

**If code is 0, do not recommend `/docs-sync`.** Recommending work that does not exist is
how people stop trusting `/docs-status`.

#### 2-5

Stale and unverified pages, unprocessed `raw/` material, unfinished `- [ ]` items, the
last 3 entries of `docs/log.md`.

```bash
python3 .claude/scripts/docs-lint.py --root . --quiet
```

#### 6. Size

Page count per category.

#### Reporting

Fit it on one screen. If something needs doing, recommend it in one line at the end.

### `/docs-lint` -- integrity check for the docs wiki

Run the script for anything a script can decide.

```bash
python3 .claude/scripts/docs-lint.py --root .
```

Catches: broken `[[links]]`, orphan pages, broken sources (`code:` paths),
missing frontmatter, citations past end of file, unfinished checklists,
**stale**, **unverified**.

##### stale and unverified are different

| | Meaning | What to do |
|---|---|---|
| **stale** | the code this page points at **changed** | `/docs-sync` |
| **unverified** | freshness **cannot be decided** | a human has to read it |

Three ways a page becomes unverified:

1. **No `code:` sources** -- nothing to compare against, so staleness is not computable.
   Legitimate for concept and history pages, which is why this never fails the run
2. **All `code:` sources have disappeared**
3. **The sha moved past N code commits with no body change** -- allowed by the schema,
   but `verified_at` is a declaration rather than proof, and this is the one place where
   "did you actually read it?" shows up. The larger N is, the more suspect

> A `verified_at` sha that is **not in the repository** (typo, rebase, force-push) is a
> **problem**, not merely unverified -- it is a broken claim, not a weak one.

#### What the script cannot catch -- read and judge these yourself

1. **Contradictions between pages** -- the same fact stated differently
2. **Missing cross-references** -- two pages covering one concept without linking
3. **Duplicated prose** -- a lookup page and an exploratory page each carrying the same
   table. That breaks the single-source rule
4. **Prose that went stale without the code moving** -- `verified_at` is current but the
   text no longer matches reality. A script cannot see this

#### Reporting

Summarise the findings and **ask the user before fixing.** Do not auto-fix.
If it is clean, say so and stop.

### `/docs-query` -- ask the docs wiki

The question is whatever the user asked.

#### Rules

1. **Answer from `docs/` only.** Get your bearings from `docs/index.md`, then read the
   relevant pages
2. **Attribute every claim with `[[page-name]]`**
3. **If the wiki does not have it, say so.**
   Do not dig through the code and improvise an answer to paper over the gap. That hides
   the fact that the wiki is empty there, and the next person hits the same wall:
   > Not in the wiki. It looks like it lives near `<path>` --
   > shall I check and add it to [[that-page]]?
4. If the answer produced a useful new synthesis, save it as a page **after confirming**

#### Reading order

- "why is it like this" -> exploratory (`concepts/`, `decisions/`)
- "how does it work" -> exploratory (`architecture/`, `subsystems/`)
- "what was the value" -> lookup (`reference/`)
- "how do I do it" -> lookup (`operations/`)
