# Security policy

HeronLoom runs entirely on your own machine. The pipeline, the dashboard,
and the search engine read and write local files. The only network calls the
project itself makes are the ones you configure: your Ollama endpoint, and
your chosen LLM API endpoint.

## Known risk area: the dashboard

`dashboard.py` has **no authentication layer**. Bound to `127.0.0.1` (the
default) it is only reachable from your own machine. If you pass
`--host 0.0.0.0` to expose it on your network, anyone on that network gets
full access to your runs, the file browser, and an interactive terminal
capable of launching processes. Only do this on a network you trust — see
[DASHBOARD.md](docs/DASHBOARD.md) for the `--allow-origin` flag it also
requires.

## License obligations when running as a network service

This is a legal point, separate from the authentication question above.
HeronLoom is licensed under the AGPL-3.0. If you deploy a **modified**
version of the dashboard so that people other than yourself interact with it
remotely over a network, the license requires you to offer those users the
Corresponding Source of your modified version — typically a "Source" link in
the interface pointing to your fork (see [LICENSE](LICENSE), section 13).
Running an unmodified copy for yourself, even on `0.0.0.0`, does not trigger
this — it applies once you've changed the code and let others use that
changed version remotely.

## Untrusted input

Retrieved post text is treated as untrusted data by `search.py`: it is fenced
inside `<retrieved_evidence>`, every line is quoted, and both the system and
user prompts instruct the model never to follow instructions found in it. That
is a mitigation, not a guarantee. Treat any report generated from an untrusted
corpus as untrusted output, particularly before pasting it somewhere that
executes or renders it.

## Logs

Logs (`logs/`) may include excerpts of your input data — post text, cluster
labels, error payloads — for debugging. They stay on your machine and are
never transmitted anywhere by the tool itself, and `.gitignore` excludes the
directory. See [ARCHITECTURE.md](docs/ARCHITECTURE.md#logs) for rotation and
location.

## Reporting a vulnerability

Please do not open a public GitHub issue for a security vulnerability. Use
this repository's Security tab and its **Report a vulnerability** button
(GitHub's private vulnerability reporting) instead.
