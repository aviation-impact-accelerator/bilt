# bilt-proposals

Templates of proposed Bilt APIs. Each private module here is one proposal, re-exported from `bilt_proposals` the way it would eventually be exported from `bilt`. A proposal defines the names and signatures an example would import, so that examples written against the proposal can be imported, linted and type-checked. Nothing new is implemented. Using a proposed feature raises `ProposalNotImplementedError`, so tests that rely on a proposal can be marked `xfail(raises=ProposalNotImplementedError)`.

| Module | Names | Proposal |
|---|---|---|
| `_connectors.py` | `field(flow=True)`, `connection` | Interface-typed fields as groups of ports; flow ports; `@connection` to join groups at a junction |

When a proposal is built in Bilt, its module here is deleted, examples change `from bilt_proposals import ...` to `from bilt import ...`, and the `xfail` markers are removed (`xfail_strict = true` makes the tests fail until that happens).
