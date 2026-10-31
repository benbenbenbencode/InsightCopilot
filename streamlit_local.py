"""Compatibility entry point.

Earlier versions shipped a second, API-free UI here. Both entry points now run
the same dashboard, which calls the pipeline in-process and needs no server:

    streamlit run streamlit_local.py
    streamlit run streamlit_app.py
"""

from app.dashboard import main

main()
