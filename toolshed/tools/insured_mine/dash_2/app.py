"""
insuredmind_dash.app
Root launcher that wires the sidebar, URL routing and global assets.

Style: 79‑char lines, Google‑style docstrings, Black‑formatted.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

# -----------------------------------------------------------------------------
# Dash initialisation – identical to the Amazon style guide
# -----------------------------------------------------------------------------
app: Dash = Dash(
    __name__,
    use_pages=True,                 # <– enables multi‑page routing
    title="We Insure Dashboard",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server                # gunicorn entry‑point

# -----------------------------------------------------------------------------
# Sidebar (left‑hand navigation)
# -----------------------------------------------------------------------------
_SIDEBAR = html.Div(
    [
        # Company logo
        dbc.Row(
            html.Img(src="assets/logos/we_insure.svg", height=35),
            className="sidebar-logo",
        ),
        html.Hr(),
        # Page links (must match pages/<page>.py  register_page(path=...)
        dbc.Nav(
            [
                dbc.NavLink("Overview",          href="/",                active="exact"),
                dbc.NavLink("Agent performance", href="/kpi_by_agent",    active="exact"),
                dbc.NavLink("Pipeline speed",    href="/pipeline_speed",  active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        # Footer inside the sidebar
        html.Div(
            [
                html.Span("Built by We Insure Analytics • "),
                html.A("Source", href="https://github.com/your‑org/insuredmind_dash", target="_blank"),
            ],
            className="subtitle-sidebar",
            style={"position": "absolute", "bottom": "10px", "width": "100%"},
        ),
    ],
    className="sidebar",
)

# -----------------------------------------------------------------------------
# Page container (Dash injects the active page into dash.page_container)
# -----------------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url"),    # keeps browser URL in sync
        _SIDEBAR,
        html.Div(dash.page_container, className="page-content"),
    ]
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Dash ≥ 3.0 API – (Note: app.run_server is obsolete)
    app.run(debug=True, port=8050)
