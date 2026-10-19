"""Training / sampling / evaluation pipeline for LLM Flow Processes.

These modules depend on `core` (this package's flattened library) for the flow-matching subset: `FlowProcess`,
`FlowMatchingTransformer`, `flow_matching`, `conditioning`, `sampling`.

Entry points (each is a click CLI):

    train       train a flow-matching NDP prior
    pipeline    sample NDP / NDP+LLM, build LLM score caches
    llm_direct  LLM-Direct ensembles

Run one with either form:

    uv run python neural_diffusion_processes/llm_direct.py --help
    uv run python -m neural_diffusion_processes.llm_direct --help

`pipeline` is also the package's import surface: it re-exports the presets,
the LLM score machinery and the plotting helpers, so consumers import it
rather than reaching into individual modules.

    from neural_diffusion_processes import pipeline as ndp
"""
