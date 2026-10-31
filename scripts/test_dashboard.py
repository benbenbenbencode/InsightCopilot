"""Headless smoke test for the Streamlit dashboard.

Runs the real script through Streamlit's AppTest harness, so exceptions that
only appear at render time (widget config, layout, missing keys) surface here
instead of in the browser.
"""

import sys

from streamlit.testing.v1 import AppTest


def main() -> int:
    at = AppTest.from_file("streamlit_app.py", default_timeout=180)
    at.run()

    if at.exception:
        print("EXCEPTION during initial render:")
        for item in at.exception:
            print(item.value)
        return 1

    print("initial render OK")
    print(f"  tabs: {len(at.tabs)}  buttons: {len(at.button)}  text areas: {len(at.text_area)}")

    # Ask a question through the UI.
    at.text_area[0].input("How did daily revenue trend over the last 30 days?")
    at.button[0].click()
    at.run()

    if at.exception:
        print("EXCEPTION during analyse:")
        for item in at.exception:
            print(item.value)
        return 1

    print("analyse OK")
    print(f"  errors shown: {len(at.error)}")
    print(f"  dataframes: {len(at.dataframe)}")
    print(f"  subheaders: {[s.value for s in at.subheader][:3]}")
    print(f"  captions: {len(at.caption)}")

    if at.error:
        for item in at.error:
            print("  error:", item.value)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
