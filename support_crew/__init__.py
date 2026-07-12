"""Multi-Agent Customer Support System — application package.

Modules:
    config     -- environment loading, settings and validation helpers
    models     -- pydantic / typed models shared across the app
    errors     -- error taxonomy, user-safe messages, retry policy
    storage    -- answers.txt persistence (locked appends, IDs, rotation)
    tools      -- CrewAI tools (file-saving tool for the Entry Agent)
    crew       -- agent/task/crew construction and execution
    ui         -- Streamlit presentation layer (views, components, styles)
"""
