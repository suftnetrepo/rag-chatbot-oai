"""
RAG Chatbot Dash UI — clean, reliable implementation.
Typewriter: dcc.Interval polls dcc.Store, reveals text word-by-word via dcc.Markdown.
No JS tricks. No clientside callbacks. Pure Dash.
"""
from __future__ import annotations
import base64
import requests
from dash import Dash, Input, Output, State, dcc, html, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

API_BASE = "http://localhost:8000/api"

C = {
    "base":       "#0A0B0F",
    "surface":    "#111318",
    "raised":     "#1C1E27",
    "border":     "#2A2D3A",
    "accent":     "#5B7FFF",
    "accent_dim": "#3D5CE8",
    "text":       "#E4E6F0",
    "muted":      "#6B6F85",
    "faint":      "#3A3D52",
    "green":      "#4ADE80",
    "red":        "#F87171",
    "amber":      "#FBBF24",
}

WORDS_PER_TICK = 4    # words revealed per tick
TICK_MS        = 60   # interval between ticks → smooth word-by-word


# ── UI helpers ─────────────────────────────────────────────────────────────

def avatar(label, bg):
    return html.Div(label, style={
        "width": "30px", "height": "30px", "borderRadius": "9px",
        "background": bg, "display": "flex", "alignItems": "center",
        "justifyContent": "center", "fontSize": "14px", "flexShrink": "0",
        "border": f"1px solid {C['border']}",
    })


def user_bubble(text):
    return html.Div(
        style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "20px"},
        children=[html.Div(
            style={"display": "flex", "alignItems": "flex-start",
                   "gap": "8px", "flexDirection": "row-reverse"},
            children=[
                avatar("U", C["accent_dim"]),
                html.Div(
                    html.P(text, style={"margin": 0, "fontSize": "14px",
                                        "lineHeight": "1.65", "color": "#fff",
                                        "whiteSpace": "pre-wrap"}),
                    style={"background": C["accent"],
                           "borderRadius": "14px 14px 4px 14px",
                           "padding": "12px 16px", "maxWidth": "68%"},
                ),
            ],
        )],
    )


def bot_bubble(text, sources=None, latency_ms=None, streaming=False):
    seen, chips = set(), []
    for s in (sources or []):
        key = f"{s.get('filename','')}:{s.get('chunk_index',0)}"
        if key in seen:
            continue
        seen.add(key)
        chips.append(html.Span(
            f"📎 {s.get('filename','?')} · chunk {s.get('chunk_index',0)} · {s.get('relevance_score',0):.2f}",
            style={
                "display": "inline-block", "fontSize": "10px",
                "color": C["accent"], "background": "rgba(91,127,255,0.1)",
                "border": "1px solid rgba(91,127,255,0.22)",
                "borderRadius": "20px", "padding": "2px 10px",
                "marginRight": "4px", "marginTop": "4px",
            },
        ))

    # Append blinking cursor while streaming
    display_text = text + (" ▋" if streaming else "")

    children = [
        dcc.Markdown(display_text,
                     style={"margin": 0, "fontSize": "14px",
                            "lineHeight": "1.7", "color": C["text"]}),
    ]
    if chips and not streaming:
        children.append(html.Div(chips, style={"marginTop": "8px"}))
    if latency_ms and not streaming:
        children.append(html.Span(
            f"⏱ {latency_ms} ms",
            style={"fontSize": "10px", "color": C["faint"],
                   "display": "block", "marginTop": "6px"},
        ))

    return html.Div(
        style={"display": "flex", "justifyContent": "flex-start",
               "marginBottom": "20px"},
        children=[html.Div(
            style={"display": "flex", "alignItems": "flex-start", "gap": "8px"},
            children=[
                avatar("🦙", "rgba(91,127,255,0.15)"),
                html.Div(children,
                         style={"background": C["raised"],
                                "border": f"1px solid {C['border']}",
                                "borderRadius": "14px 14px 14px 4px",
                                "padding": "13px 16px", "maxWidth": "72%"}),
            ],
        )],
    )


def thinking_bubble():
    return html.Div(
        style={"display": "flex", "justifyContent": "flex-start",
               "marginBottom": "20px"},
        children=[html.Div(
            style={"display": "flex", "alignItems": "flex-start", "gap": "8px"},
            children=[
                avatar("🦙", "rgba(91,127,255,0.15)"),
                html.Div(
                    style={"background": C["raised"],
                           "border": f"1px solid {C['border']}",
                           "borderRadius": "14px 14px 14px 4px",
                           "padding": "14px 18px", "display": "flex",
                           "gap": "5px", "alignItems": "center"},
                    children=[
                        html.Span(className="dot-1",
                                  style={"width": "7px", "height": "7px",
                                         "borderRadius": "50%", "background": C["muted"],
                                         "display": "inline-block"}),
                        html.Span(className="dot-2",
                                  style={"width": "7px", "height": "7px",
                                         "borderRadius": "50%", "background": C["muted"],
                                         "display": "inline-block"}),
                        html.Span(className="dot-3",
                                  style={"width": "7px", "height": "7px",
                                         "borderRadius": "50%", "background": C["muted"],
                                         "display": "inline-block"}),
                    ],
                ),
            ],
        )],
    )


def empty_state():
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "alignItems": "center",
               "justifyContent": "center", "height": "100%", "gap": "16px"},
        children=[
            html.Div("🦙", style={"fontSize": "48px"}),
            html.H3("How can I help you?",
                    style={"color": C["text"], "fontSize": "18px",
                           "fontWeight": "600", "margin": 0}),
            html.P("Upload a document in the sidebar, then ask me anything about it.",
                   style={"fontSize": "13px", "color": C["muted"],
                          "maxWidth": "300px", "textAlign": "center",
                          "lineHeight": "1.65", "margin": 0}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gap": "8px", "marginTop": "6px",
                       "width": "100%", "maxWidth": "400px"},
                children=[
                    html.Div(p, style={
                        "padding": "10px 14px", "borderRadius": "10px",
                        "border": f"1px solid {C['border']}",
                        "background": C["raised"], "fontSize": "12px",
                        "color": C["muted"], "lineHeight": "1.5",
                    })
                    for p in ["Summarise this document", "What are the key skills?",
                              "List the main experience", "What technologies are used?"]
                ],
            ),
        ],
    )


def build_feed(messages, tw_text=None, tw_sources=None, tw_latency=None):
    """Render all message bubbles. tw_* args control the streaming bubble."""
    bubbles = []
    for i, m in enumerate(messages):
        is_last = i == len(messages) - 1
        if m["role"] == "user":
            bubbles.append(user_bubble(m["content"]))
        elif is_last and tw_text is not None:
            # Streaming: show partial text with cursor
            full    = m["content"]
            words   = full.split(" ")
            visible = " ".join(words[:tw_text]) if tw_text < len(words) else full
            done    = tw_text >= len(words)
            bubbles.append(bot_bubble(
                visible,
                sources=tw_sources if done else None,
                latency_ms=tw_latency if done else None,
                streaming=not done,
            ))
        else:
            bubbles.append(bot_bubble(
                m["content"], m.get("sources"), m.get("latency_ms")
            ))
    return bubbles or [empty_state()]


# ── App ────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="RAG Chatbot",
    update_title=None,
)

app.index_string = """<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { overflow: hidden; background: #0A0B0F; color: #E4E6F0;
           font-family: Inter, system-ui, sans-serif;
           -webkit-font-smoothing: antialiased; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2A2D3A; border-radius: 2px; }
    textarea:focus { outline: 2px solid rgba(91,127,255,0.4) !important;
                     border-color: #5B7FFF !important; }
    @keyframes dotBounce {
      0%,80%,100% { transform:translateY(0); opacity:0.3; }
      40%          { transform:translateY(-5px); opacity:1; }
    }
    .dot-1 { animation: dotBounce 1.2s infinite 0.0s; }
    .dot-2 { animation: dotBounce 1.2s infinite 0.2s; }
    .dot-3 { animation: dotBounce 1.2s infinite 0.4s; }
    .send-btn:hover  { background: #3D5CE8 !important; }
    .send-btn:active { transform: scale(0.96); }
    .upload-zone:hover { border-color: #5B7FFF !important;
                         background: rgba(91,127,255,0.04) !important; }
    .session-item:hover { background: #1C1E27 !important; }
    #feed { scroll-behavior: smooth; }
  </style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
  /* Auto-scroll feed */
  var _fo = new MutationObserver(function() {
    var f = document.getElementById('feed');
    if (f) f.scrollTop = f.scrollHeight;
  });
  document.addEventListener('DOMContentLoaded', function() {
    var f = document.getElementById('feed');
    if (f) _fo.observe(f, { childList: true, subtree: true });
  });
</script>
</body>
</html>"""

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column",
           "height": "100vh", "background": C["base"], "overflow": "hidden"},
    children=[

        # Header
        html.Div(
            style={"display": "flex", "alignItems": "center",
                   "justifyContent": "space-between", "padding": "0 24px",
                   "height": "52px", "background": C["surface"],
                   "borderBottom": f"1px solid {C['border']}", "flexShrink": "0"},
            children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"},
                         children=[
                             html.Div("🦙", style={
                                 "width": "30px", "height": "30px", "borderRadius": "8px",
                                 "background": C["accent"], "display": "flex",
                                 "alignItems": "center", "justifyContent": "center",
                                 "fontSize": "15px",
                             }),
                             html.Span("RAG Chatbot",
                                       style={"fontWeight": "600", "fontSize": "15px",
                                              "letterSpacing": "-0.3px"}),
                             html.Span("OpenAI · LlamaIndex · Typesense", style={
                                 "fontSize": "11px", "color": C["muted"],
                                 "marginLeft": "10px", "paddingLeft": "12px",
                                 "borderLeft": f"1px solid {C['border']}",
                             }),
                         ]),
                html.Div(id="health-bar"),
            ],
        ),

        # Body
        html.Div(
            style={"display": "flex", "flex": "1", "overflow": "hidden"},
            children=[

                # Sidebar
                html.Div(
                    style={"width": "256px", "flexShrink": "0",
                           "background": C["surface"],
                           "borderRight": f"1px solid {C['border']}",
                           "display": "flex", "flexDirection": "column",
                           "padding": "14px 10px", "gap": "14px", "overflowY": "auto"},
                    children=[
                        html.Button("+ New conversation", id="btn-new", n_clicks=0,
                                    style={"width": "100%", "padding": "10px 14px",
                                           "background": "rgba(91,127,255,0.12)",
                                           "border": "1px solid rgba(91,127,255,0.3)",
                                           "borderRadius": "10px", "color": C["accent"],
                                           "fontSize": "13px", "fontWeight": "500",
                                           "cursor": "pointer", "transition": "all 0.15s"}),
                        html.Div([
                            html.P("Conversations",
                                   style={"fontSize": "10px", "fontWeight": "600",
                                          "color": C["muted"], "marginBottom": "6px",
                                          "paddingLeft": "4px"}),
                            html.Div(id="session-list"),
                        ]),
                        html.Hr(style={"borderColor": C["border"], "margin": "0"}),
                        html.Div([
                            html.P("Knowledge base",
                                   style={"fontSize": "10px", "fontWeight": "600",
                                          "color": C["muted"], "marginBottom": "8px",
                                          "paddingLeft": "4px"}),
                            dcc.Upload(
                                id="uploader", multiple=True,
                                className="upload-zone",
                                children=html.Div([
                                    html.Div("📄", style={"fontSize": "20px",
                                                           "marginBottom": "4px"}),
                                    html.P("Drop files or click to upload",
                                           style={"fontSize": "12px",
                                                  "color": C["muted"], "margin": 0}),
                                    html.P("PDF · TXT · MD · CSV · DOCX",
                                           style={"fontSize": "10px",
                                                  "color": C["faint"], "margin": "3px 0 0"}),
                                ], style={"textAlign": "center"}),
                                style={"border": f"1.5px dashed {C['border']}",
                                       "borderRadius": "10px", "padding": "16px 10px",
                                       "cursor": "pointer", "marginBottom": "8px",
                                       "transition": "all 0.15s"},
                            ),
                            html.Div(id="upload-toast"),
                            html.Div(id="doc-list"),
                        ]),
                    ],
                ),

                # Main
                html.Div(
                    style={"flex": "1", "display": "flex",
                           "flexDirection": "column", "minWidth": "0"},
                    children=[
                        dcc.Tabs(
                            id="tabs", value="chat",
                            colors={"border": C["border"], "primary": C["accent"],
                                    "background": C["surface"]},
                            style={"flexShrink": "0",
                                   "borderBottom": f"1px solid {C['border']}",
                                   "background": C["surface"]},
                            children=[
                                dcc.Tab(label="Chat", value="chat",
                                        style={"color": C["muted"],
                                               "background": C["surface"],
                                               "border": "none", "padding": "12px 20px",
                                               "fontSize": "13px"},
                                        selected_style={"color": C["text"],
                                                        "background": C["surface"],
                                                        "borderBottom": f"2px solid {C['accent']}",
                                                        "border": "none",
                                                        "padding": "12px 20px",
                                                        "fontSize": "13px",
                                                        "fontWeight": "500"}),
                                dcc.Tab(label="Analytics", value="analytics",
                                        style={"color": C["muted"],
                                               "background": C["surface"],
                                               "border": "none", "padding": "12px 20px",
                                               "fontSize": "13px"},
                                        selected_style={"color": C["text"],
                                                        "background": C["surface"],
                                                        "borderBottom": f"2px solid {C['accent']}",
                                                        "border": "none",
                                                        "padding": "12px 20px",
                                                        "fontSize": "13px",
                                                        "fontWeight": "500"}),
                            ],
                        ),

                        # Chat panel
                        html.Div(
                            id="chat-wrap",
                            style={"flex": "1", "display": "flex",
                                   "flexDirection": "column", "overflow": "hidden"},
                            children=[
                                html.Div(id="feed",
                                         style={"flex": "1", "overflowY": "auto",
                                                "padding": "28px 32px"},
                                         children=[empty_state()]),
                                html.Div(
                                    style={"padding": "12px 20px 16px",
                                           "background": C["surface"],
                                           "borderTop": f"1px solid {C['border']}",
                                           "flexShrink": "0"},
                                    children=[
                                        html.Div(
                                            style={"display": "flex", "gap": "10px",
                                                   "alignItems": "flex-end"},
                                            children=[
                                                dcc.Textarea(
                                                    id="input",
                                                    placeholder="Ask anything… (Enter to send · Shift+Enter for new line)",
                                                    rows=1,
                                                    style={
                                                        "flex": "1",
                                                        "background": C["raised"],
                                                        "border": f"1px solid {C['border']}",
                                                        "borderRadius": "12px",
                                                        "color": C["text"],
                                                        "fontSize": "14px",
                                                        "padding": "11px 16px",
                                                        "outline": "none",
                                                        "resize": "none",
                                                        "minHeight": "44px",
                                                        "maxHeight": "160px",
                                                        "fontFamily": "inherit",
                                                        "lineHeight": "1.5",
                                                        "transition": "border-color 0.15s",
                                                    },
                                                ),
                                                html.Button(
                                                    "Send ↵", id="btn-send",
                                                    n_clicks=0, className="send-btn",
                                                    style={
                                                        "padding": "11px 20px",
                                                        "background": C["accent"],
                                                        "border": "none",
                                                        "borderRadius": "12px",
                                                        "color": "#fff",
                                                        "fontSize": "13px",
                                                        "fontWeight": "500",
                                                        "cursor": "pointer",
                                                        "whiteSpace": "nowrap",
                                                        "transition": "all 0.15s",
                                                        "flexShrink": "0",
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.P(
                                            "Powered by OpenAI GPT-4o · LlamaIndex · Typesense · PostgreSQL",
                                            style={"fontSize": "10px", "color": C["faint"],
                                                   "textAlign": "center", "marginTop": "8px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(id="analytics-wrap",
                                 style={"display": "none", "flex": "1",
                                        "overflowY": "auto", "padding": "24px 32px"}),
                    ],
                ),
            ],
        ),

        # Stores + intervals
        dcc.Store(id="session-id",  storage_type="session"),
        dcc.Store(id="messages",    data=[]),
        dcc.Store(id="tw-store",    data={"active": False, "word_pos": 0,
                                          "sources": [], "latency_ms": None}),
        dcc.Interval(id="tw-tick",      interval=TICK_MS, n_intervals=0, disabled=True),
        dcc.Interval(id="tick-health",  interval=15_000,  n_intervals=0),
        dcc.Interval(id="tick-sidebar", interval=6_000,   n_intervals=0),
    ],
)


# ── Health ─────────────────────────────────────────────────────────────────

@app.callback(Output("health-bar", "children"),
              Input("tick-health", "n_intervals"))
def refresh_health(_):
    try:
        svcs = requests.get(f"{API_BASE}/health", timeout=3).json().get("services", {})
        return html.Div(
            style={"display": "flex", "gap": "14px", "alignItems": "center"},
            children=[
                html.Span([
                    html.Span(style={
                        "display": "inline-block", "width": "7px", "height": "7px",
                        "borderRadius": "50%", "marginRight": "5px",
                        "background": C["green"] if ok else C["red"],
                        "verticalAlign": "middle",
                    }),
                    html.Span(svc, style={"fontSize": "11px", "color": C["muted"]}),
                ])
                for svc, ok in svcs.items()
            ])
    except Exception:
        return html.Span("⚠ API offline", style={"fontSize": "11px", "color": C["red"]})


# ── Tab switch ─────────────────────────────────────────────────────────────

@app.callback(
    Output("chat-wrap",      "style"),
    Output("analytics-wrap", "style"),
    Output("analytics-wrap", "children"),
    Input("tabs", "value"),
)
def switch_tab(tab):
    show_chat = {"flex": "1", "display": "flex",
                 "flexDirection": "column", "overflow": "hidden"}
    show_anal = {"flex": "1", "overflowY": "auto", "padding": "24px 32px"}
    hide      = {"display": "none"}
    if tab == "analytics":
        return hide, show_anal, _analytics_content()
    return show_chat, hide, []


def _analytics_content():
    try:
        s = requests.get(f"{API_BASE}/stats", timeout=3).json()
    except Exception:
        s = {}
    kpis = [
        ("Sessions",    s.get("total_sessions",  "—")),
        ("Messages",    s.get("total_messages",  "—")),
        ("Documents",   s.get("total_documents", "—")),
        ("Chunks",      s.get("total_chunks",    "—")),
        ("Avg latency", f"{s.get('avg_latency_ms',0):.0f} ms"
         if s.get("avg_latency_ms") else "—"),
    ]
    fig = go.Figure(go.Scatter(
        y=[320, 480, 290, 560, 310, 420, 380, 470, 290, 350],
        mode="lines+markers",
        line=dict(color=C["accent"], width=2),
        marker=dict(size=4, color=C["accent"]),
        fill="tozeroy", fillcolor="rgba(91,127,255,0.08)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=200,
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, color=C["muted"]),
        yaxis=dict(showgrid=True, gridcolor=C["border"],
                   zeroline=False, color=C["muted"], ticksuffix=" ms"),
    )
    return [
        html.H3("Analytics", style={"fontSize": "16px", "fontWeight": "600",
                                     "color": C["text"], "marginBottom": "20px"}),
        html.Div(
            style={"display": "flex", "gap": "12px",
                   "flexWrap": "wrap", "marginBottom": "20px"},
            children=[
                html.Div(
                    style={"background": C["surface"],
                           "border": f"1px solid {C['border']}",
                           "borderRadius": "12px", "padding": "16px",
                           "minWidth": "120px", "flex": "1"},
                    children=[
                        html.P(label, style={"fontSize": "11px", "color": C["muted"],
                                              "marginBottom": "6px"}),
                        html.H3(str(val), style={"fontSize": "22px",
                                                  "fontWeight": "600",
                                                  "color": C["text"]}),
                    ],
                )
                for label, val in kpis
            ],
        ),
        html.Div(
            style={"background": C["surface"],
                   "border": f"1px solid {C['border']}",
                   "borderRadius": "12px", "padding": "16px"},
            children=[
                html.P("Response latency (ms)",
                       style={"fontSize": "12px", "color": C["muted"],
                              "marginBottom": "10px"}),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ],
        ),
    ]


# ── Step 1: Send message → show thinking → call API → start typewriter ─────

@app.callback(
    Output("feed",       "children",  allow_duplicate=True),
    Output("messages",   "data",      allow_duplicate=True),
    Output("session-id", "data",      allow_duplicate=True),
    Output("input",      "value"),
    Output("btn-send",   "disabled"),
    Output("tw-store",   "data",      allow_duplicate=True),
    Output("tw-tick",    "disabled",  allow_duplicate=True),
    Input("btn-send",    "n_clicks"),
    Input("btn-new",     "n_clicks"),
    State("input",       "value"),
    State("messages",    "data"),
    State("session-id",  "data"),
    prevent_initial_call=True,
)
def send_message(send_n, new_n, text, messages, session_id):
    triggered = callback_context.triggered[0]["prop_id"]

    if "btn-new" in triggered:
        blank = {"active": False, "word_pos": 0, "sources": [], "latency_ms": None}
        return [empty_state()], [], None, "", False, blank, True

    if "btn-send" not in triggered or not text or not text.strip():
        raise PreventUpdate

    messages  = list(messages or [])
    user_text = text.strip()
    messages.append({"role": "user", "content": user_text,
                     "sources": [], "latency_ms": None})

    # Show thinking immediately
    bubbles = [user_bubble(m["content"]) if m["role"] == "user"
               else bot_bubble(m["content"], m.get("sources"), m.get("latency_ms"))
               for m in messages[:-1]]
    bubbles.append(user_bubble(user_text))
    bubbles.append(thinking_bubble())

    # Call API
    try:
        payload = {"message": user_text}
        if session_id:
            payload["session_id"] = session_id
        r = requests.post(f"{API_BASE}/chat", json=payload, timeout=90)
        r.raise_for_status()
        data       = r.json()
        answer     = data["answer"]
        sources    = data.get("sources", [])
        latency_ms = data.get("latency_ms")
        session_id = data.get("session_id", session_id)
    except requests.exceptions.Timeout:
        answer, sources, latency_ms = "⚠️ Request timed out. Please try again.", [], None
    except Exception as e:
        answer, sources, latency_ms = f"⚠️ Error: {str(e)}", [], None

    messages.append({"role": "assistant", "content": answer,
                     "sources": sources, "latency_ms": latency_ms})

    # Deduplicate sources by filename:chunk
    seen, clean_sources = set(), []
    for s in sources:
        key = f"{s.get('filename','')}:{s.get('chunk_index',0)}"
        if key not in seen:
            seen.add(key)
            clean_sources.append(s)

    # Start typewriter from word 0
    tw = {"active": True, "word_pos": 0,
          "sources": clean_sources, "latency_ms": latency_ms}

    # Initial feed: all previous + empty streaming bubble
    bubbles = [user_bubble(m["content"]) if m["role"] == "user"
               else bot_bubble(m["content"], m.get("sources"), m.get("latency_ms"))
               for m in messages[:-1]]
    bubbles.append(bot_bubble("▋", streaming=True))

    # tw-tick disabled=False starts the interval fresh
    # btn-send disabled=True until typewriter finishes
    return bubbles, messages, session_id, "", True, tw, False


# ── Step 2: Typewriter tick ────────────────────────────────────────────────

@app.callback(
    Output("feed",     "children",  allow_duplicate=True),
    Output("tw-store", "data",      allow_duplicate=True),
    Output("tw-tick",  "disabled",  allow_duplicate=True),
    Output("btn-send", "disabled",  allow_duplicate=True),
    Input("tw-tick",   "n_intervals"),
    State("tw-store",  "data"),
    State("messages",  "data"),
    prevent_initial_call=True,
)
def typewriter_tick(_, tw, messages):
    if not tw or not tw.get("active"):
        raise PreventUpdate

    messages = list(messages or [])
    if not messages:
        raise PreventUpdate

    last = messages[-1]
    if last["role"] != "assistant":
        raise PreventUpdate

    full_text  = last["content"]
    words      = full_text.split(" ")
    total      = len(words)
    pos        = tw.get("word_pos", 0)
    new_pos    = min(pos + WORDS_PER_TICK, total)
    done       = new_pos >= total
    revealed   = " ".join(words[:new_pos])

    sources    = tw.get("sources", [])
    latency_ms = tw.get("latency_ms")

    # Rebuild feed
    bubbles = [user_bubble(m["content"]) if m["role"] == "user"
               else bot_bubble(m["content"], m.get("sources"), m.get("latency_ms"))
               for m in messages[:-1]]

    bubbles.append(bot_bubble(
        revealed,
        sources=sources if done else None,
        latency_ms=latency_ms if done else None,
        streaming=not done,
    ))

    new_tw = {**tw, "word_pos": new_pos, "active": not done}
    # done=True  → disable interval, enable send button
    # done=False → keep interval running, keep send disabled
    return bubbles, new_tw, done, done


# ── Upload + docs ──────────────────────────────────────────────────────────

@app.callback(
    Output("upload-toast", "children"),
    Output("doc-list",     "children"),
    Input("uploader",      "contents"),
    Input("tick-sidebar",  "n_intervals"),
    State("uploader",      "filename"),
    prevent_initial_call=False,
)
def handle_upload_and_docs(contents, _, filenames):
    triggered = (callback_context.triggered[0]["prop_id"]
                 if callback_context.triggered else "")
    toasts = []

    if "uploader" in triggered and contents:
        for content, fname in zip(contents, filenames):
            _, encoded = content.split(",", 1)
            try:
                r = requests.post(
                    f"{API_BASE}/documents/upload",
                    files={"file": (fname, base64.b64decode(encoded))},
                    timeout=120,
                )
                r.raise_for_status()
                d  = r.json()
                ok = d.get("status") == "indexed"
                toasts.append(html.Div(
                    [html.Span("✅ " if ok else "❌ "), d.get("message", fname)],
                    style={
                        "fontSize": "11px", "padding": "7px 10px",
                        "borderRadius": "8px", "marginBottom": "4px",
                        "background": ("rgba(74,222,128,0.08)" if ok
                                       else "rgba(248,113,113,0.08)"),
                        "border": (f"1px solid {'rgba(74,222,128,0.25)' if ok else 'rgba(248,113,113,0.25)'}"),
                        "color": C["green"] if ok else C["red"],
                    },
                ))
            except Exception as e:
                toasts.append(html.Div(
                    f"❌ {fname}: {str(e)[:60]}",
                    style={"fontSize": "11px", "padding": "7px 10px",
                           "borderRadius": "8px", "marginBottom": "4px",
                           "background": "rgba(248,113,113,0.08)",
                           "border": "1px solid rgba(248,113,113,0.25)",
                           "color": C["red"]},
                ))

    ext_icon = {"pdf": "📕", "md": "📝", "txt": "📃", "csv": "📊", "docx": "📄"}
    try:
        docs = requests.get(f"{API_BASE}/documents", timeout=3).json()
    except Exception:
        docs = []

    doc_items = html.Div([
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "7px",
                   "padding": "7px 10px", "borderRadius": "8px", "marginBottom": "4px",
                   "background": C["raised"], "border": f"1px solid {C['border']}"},
            children=[
                html.Span(ext_icon.get(d.get("file_type", ""), "📄"),
                          style={"fontSize": "13px"}),
                html.Span(d.get("filename", "?"),
                          style={"fontSize": "11px", "color": C["text"], "flex": "1",
                                 "overflow": "hidden", "textOverflow": "ellipsis",
                                 "whiteSpace": "nowrap"}),
                html.Span(
                    "● indexed" if d.get("status") == "indexed"
                    else f"⚠ {d.get('status', '')}",
                    style={"fontSize": "10px", "whiteSpace": "nowrap",
                           "color": (C["green"] if d.get("status") == "indexed"
                                     else C["amber"])},
                ),
            ],
        )
        for d in docs[:20]
    ]) if docs else html.P("No documents yet.",
                            style={"fontSize": "11px", "color": C["faint"],
                                   "textAlign": "center", "padding": "8px 0"})

    return toasts, doc_items


@app.callback(
    Output("session-list", "children"),
    Input("tick-sidebar",  "n_intervals"),
    Input("messages",      "data"),
    State("session-id",    "data"),
)
def refresh_sessions(_, messages, active_id):
    try:
        sessions = requests.get(f"{API_BASE}/sessions", timeout=3).json()
    except Exception:
        return html.P("—", style={"fontSize": "11px", "color": C["faint"]})

    if not sessions:
        return html.P("No conversations yet.",
                      style={"fontSize": "11px", "color": C["faint"], "padding": "4px"})

    return html.Div([
        html.Div(
            className="session-item",
            style={
                "display": "flex", "alignItems": "center", "gap": "8px",
                "padding": "7px 10px", "borderRadius": "8px", "marginBottom": "2px",
                "cursor": "pointer", "transition": "background 0.15s",
                "background": (C["raised"] if str(s.get("id")) == str(active_id)
                               else "transparent"),
                "border": (f"1px solid {C['border']}"
                           if str(s.get("id")) == str(active_id)
                           else "1px solid transparent"),
            },
            children=[
                html.Span("💬", style={"fontSize": "12px", "flexShrink": "0"}),
                html.Span(s.get("title", "Conversation"),
                          style={"fontSize": "12px", "flex": "1",
                                 "overflow": "hidden", "textOverflow": "ellipsis",
                                 "whiteSpace": "nowrap",
                                 "color": (C["text"] if str(s.get("id")) == str(active_id)
                                           else C["muted"])}),
                html.Span(str(s.get("message_count", 0)),
                          style={"fontSize": "10px", "color": C["faint"],
                                 "background": C["raised"],
                                 "padding": "1px 6px", "borderRadius": "10px"}),
            ],
        )
        for s in sessions[:20]
    ])


@app.callback(
    Output("btn-send", "disabled", allow_duplicate=True),
    Output("tw-tick",  "disabled", allow_duplicate=True),
    Input("tw-store",  "data"),
    prevent_initial_call=True,
)
def sync_controls(tw):
    """Keep Send button and interval in sync with tw-store active state."""
    if tw is None:
        return False, True
    active = tw.get("active", False)
    # If not active: enable send, disable interval
    # If active: keep send disabled, interval already enabled by send_message
    if not active:
        return False, True
    raise PreventUpdate


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
