# private/ — local-only, never committed

Everything under this directory is **git-ignored** (see the `private/**` rule in
the root `.gitignore`) because it may contain private information. Only this
README is tracked, so the location is discoverable.

Use it for local-only material that agents may read for context but that must
never be pushed:

- `transcripts/` — meeting-transcript `.txt` files. Drop them here.

Nothing you place here (outside this README) will be staged, committed, or
pushed. If you ever need a file tracked, move it out of `private/` deliberately.
