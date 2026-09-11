# Security

MCP-RagDoc is designed for trusted local use. The MCP transport is stdio and the optional HTML UI binds to loopback only.

- Source documents are read-only and constrained to the configured knowledge directory.
- Hidden, symlinked, and sensitive paths are excluded from indexing.
- Runtime SQLite/job state belongs outside Git and should not contain credentials.
- Do not expose the UI or an MCP bridge to an untrusted network without adding authentication and authorization.
- OCR vision correction is optional. By default MCP-RagDoc does not start or stop a model process; operators may point it at an already-running loopback OpenAI-compatible vision server.

Report security issues privately to the repository owner rather than opening a public issue with sensitive details.
