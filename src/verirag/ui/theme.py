"""Apple-inspired visual system for the Streamlit interface.

Adapted from Apple's WWDC design talks (*Designing Fluid Interfaces* 2018, *The
Details of UI Typography* 2020, *Principles of Great Design* 2026) as distilled in
the ``apple-design`` skill by Emil Kowalski.

What is implemented here, and why
---------------------------------
This is a **Streamlit** app, which renders server-side and hands us no pointer
event loop. That rules out a whole class of the guidance honestly rather than
faking it:

* Springs, interruptibility, velocity handoff, momentum projection and
  rubber-banding (§1-11 of the skill) all require continuous 1:1 pointer tracking
  and animation from the live presentation value. Streamlit has no such hook
  without writing a custom component, so **no spring physics is claimed here.**
  Following the skill's own default, non-gesture UI uses *critically damped*
  motion — short, ease-out, no overshoot — which is what it prescribes for
  anything the user did not throw with momentum.

Faithfully applied:

* **§12 Materials & depth** — chrome is a translucent layer (``backdrop-filter``)
  with content scrolling underneath, a bright top edge where light catches the
  material, heavier material for structural regions (sidebar) and lighter for
  interactive elements, deeper shadow on larger surfaces, and never a light
  translucent surface stacked on another.
* **§15 Typography** — the platform system font, and **size-specific tracking**:
  negative on display sizes, ~0 on body, slightly positive on captions. Leading
  varies inversely with size. Hierarchy comes from weight+size+leading as a set.
  Spacing is in ``rem`` so it scales with the user's text size.
* **§14 Reduced motion / transparency / contrast** — all three media queries are
  honoured, and reduced motion degrades to a cross-fade rather than to nothing.
* **§1 Response** — feedback lives on ``:active`` (pointer-down), not on release.
* **§11 Frame-level smoothness** — only ``transform`` and ``opacity`` animate.
* **§16 Foundations** — restraint, one accent colour, hierarchy through spacing
  and contrast rather than borders and boxes.

Selector stability
------------------
Streamlit's internal class names are generated, so every rule below targets
``data-testid`` / ``data-baseweb`` attributes, which are far more durable.
``streamlit==1.38.0`` is pinned in ``requirements.txt`` precisely so this styling
stays predictable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# design tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Motion:
    """Critically damped durations — the skill's default for non-gesture UI."""

    instant: str = "100ms"
    fast: str = "160ms"
    base: str = "240ms"
    slow: str = "360ms"
    # Apple's sheet/scroll feel: fast out of the gate, long gentle settle,
    # no overshoot. Used for reversible transitions in both directions.
    ease: str = "cubic-bezier(0.32, 0.72, 0, 1)"
    # Inverse control points, so an outbound path retraces the inbound one (§7).
    ease_in: str = "cubic-bezier(1, 0, 0.68, 0.28)"


MOTION = Motion()

TYPE_SCALE: dict[str, tuple[str, str, str, str]] = {
    # name:      (size,               line-height, letter-spacing, weight)
    "display": ("clamp(1.9rem, 3.4vw, 2.6rem)", "1.06", "-0.025em", "700"),
    "title": ("1.6rem", "1.14", "-0.021em", "660"),
    "heading": ("1.16rem", "1.28", "-0.014em", "620"),
    "body": ("0.95rem", "1.55", "0", "440"),
    "caption": ("0.78rem", "1.42", "0.006em", "500"),
}
"""Tracking is size-specific by design: a single ``letter-spacing`` is wrong
somewhere, because letters read too far apart as they grow and too tight as they
shrink.

The negative values are a little stronger than they would be for a system font,
because Inter is spaced for UI sizes and visibly loosens as it scales up. Body
stays at zero — Inter's default spacing is already tuned for exactly that size,
and tightening running text costs legibility for no gain."""


def _type_rule(selector: str, key: str) -> str:
    size, leading, tracking, weight = TYPE_SCALE[key]
    return (
        f"{selector}{{font-size:{size};line-height:{leading};"
        f"letter-spacing:{tracking};font-weight:{weight};}}"
    )


SYSTEM_FONT = (
    '"Inter", "Inter var", -apple-system, BlinkMacSystemFont, "SF Pro Text", '
    '"Segoe UI Variable Text", "Segoe UI", system-ui, Roboto, "Helvetica Neue", '
    "Arial, sans-serif"
)
"""Inter first, with the platform stack behind it.

Inter is a superset of what the system fonts do well here — a tall x-height, a
large aperture and tight, even spacing at UI sizes — and it renders identically on
every OS, which matters for a project whose screenshots and demos are the point.
The system fallback is deliberate rather than decorative: it is what renders while
the webfont is still loading, and what renders when the app runs offline, which is
a supported mode of this project.
"""

MONO_FONT = (
    '"SF Mono", ui-monospace, SFMono-Regular, "Cascadia Mono", Menlo, Consolas, monospace'
)

GOOGLE_FONTS_HREF = (
    "https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,100..900&display=swap"
)

# Inter's character variants, chosen for a document-QA interface specifically:
#   cv05 — lowercase l with a tail, so "L37" line locators cannot be misread as 1/I
#   cv08 — uppercase I with serifs, the same disambiguation from the other side
#   ss03 — round quotes and commas, which suit the softer Apple surfaces
#   calt — contextual alternates (on by default; stated so intent is explicit)
INTER_FEATURES = '"cv05" 1, "cv08" 1, "ss03" 1, "calt" 1'


def font_links() -> str:
    """Preconnect + stylesheet tags for Inter.

    Emitted as ``<link>`` rather than a CSS ``@import`` because an import inside
    an injected ``<style>`` block serialises the request behind the stylesheet
    parse, delaying first text paint. ``display=swap`` means text is readable in
    the fallback immediately and swaps to Inter when it arrives, instead of
    flashing invisible.
    """
    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link rel="stylesheet" href="{GOOGLE_FONTS_HREF}">'
    )


# ---------------------------------------------------------------------------
# stylesheet
# ---------------------------------------------------------------------------
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


def _compact(css: str) -> str:
    """Strip comments, blank lines and indentation from a stylesheet.

    This is not cosmetic — it is required for correctness. Streamlit renders
    ``st.markdown`` through a CommonMark parser, and **an HTML block ends at the
    first blank line**. A readable stylesheet with blank lines between sections
    therefore closes its own ``<style>`` tag early, and every rule after that
    point is rendered to the page as literal text. Lines indented four or more
    spaces compound it by becoming markdown code blocks.

    So the source above stays commented and readable, and what actually gets
    injected is a single unbroken run of lines with no comments and no leading
    whitespace. CSS is whitespace-insensitive, so nothing is lost.
    """
    without_comments = _COMMENT_RE.sub("", css)
    lines = (line.strip() for line in without_comments.splitlines())
    return "\n".join(line for line in lines if line)


def build_css() -> str:
    """Return the full stylesheet, compacted for safe injection."""
    return f"<style>{_compact(_STYLESHEET)}</style>"


_STYLESHEET = f"""
/* ===================================================================== */
/* Tokens. Light and dark are both first-class: colours that adapt are    */
/* part of Craft (§16), not an afterthought.                              */
/* ===================================================================== */
:root {{
  --canvas:        #f5f5f7;
  --canvas-deep:   #ececf0;
  --surface:       rgba(255, 255, 255, 0.72);
  --surface-solid: #ffffff;
  --surface-sunk:  rgba(0, 0, 0, 0.035);
  --chrome:        rgba(250, 250, 252, 0.78);
  --edge-light:    rgba(255, 255, 255, 0.65);
  --separator:     rgba(0, 0, 0, 0.08);
  --text:          #1d1d1f;
  --text-dim:      #6e6e73;
  --text-faint:    #8e8e93;

  /* Apple system colours (light). These are the semantic set from Human
     Interface Guidelines, not invented hexes — using the real values is part of
     Familiarity (§16): the palette already reads as the platform. */
  --sys-blue:   #007aff;
  --sys-green:  #34c759;
  --sys-indigo: #5856d6;
  --sys-orange: #ff9500;
  --sys-pink:   #ff2d55;
  --sys-purple: #af52de;
  --sys-red:    #ff3b30;
  --sys-teal:   #30b0c7;
  --sys-yellow: #ffcc00;
  --sys-gray:   #8e8e93;
  --sys-gray5:  #e5e5ea;
  --sys-gray6:  #f2f2f7;

  --accent:        #0071e3;   /* Apple's marketing/web blue, slightly deeper */
  --accent-hover:  #0077ed;
  --accent-soft:   rgba(0, 113, 227, 0.12);
  --green:         #1d7d43;
  --green-soft:    rgba(52, 199, 89, 0.14);
  --amber:         #8a5a00;
  --amber-soft:    rgba(255, 204, 0, 0.18);
  --red:           #b3261e;
  --red-soft:      rgba(255, 59, 48, 0.13);

  /* Larger surfaces read as thicker material: more blur, deeper shadow. */
  --blur-chip:   blur(12px) saturate(170%);
  --blur-card:   blur(20px) saturate(180%);
  --blur-chrome: blur(30px) saturate(185%);
  --shadow-chip: 0 1px 2px rgba(0,0,0,0.05), 0 2px 6px rgba(0,0,0,0.04);
  --shadow-card: 0 1px 2px rgba(0,0,0,0.05), 0 8px 24px -6px rgba(0,0,0,0.10);
  --shadow-lift: 0 2px 6px rgba(0,0,0,0.07), 0 18px 44px -12px rgba(0,0,0,0.18);

  --r-control: 10px;
  --r-card:    14px;
  --r-surface: 20px;
  --r-pill:    980px;

  --dur-instant: {MOTION.instant};
  --dur-fast:    {MOTION.fast};
  --dur-base:    {MOTION.base};
  --dur-slow:    {MOTION.slow};
  /* Mirrored pair for reversible transitions (§7): the return path uses the
     inverse control points, so out retraces in. */
  --ease:     {MOTION.ease};
  --ease-out: {MOTION.ease};
  --ease-in:  {MOTION.ease_in};
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --canvas:        #000000;
    --canvas-deep:   #0a0a0c;
    --surface:       rgba(28, 28, 30, 0.72);
    --surface-solid: #1c1c1e;
    --surface-sunk:  rgba(255, 255, 255, 0.05);
    --chrome:        rgba(22, 22, 24, 0.78);
    --edge-light:    rgba(255, 255, 255, 0.10);
    --separator:     rgba(255, 255, 255, 0.11);
    --text:          #f5f5f7;
    --text-dim:      #98989d;
    --text-faint:    #7c7c81;

    /* Apple system colours (dark) — brighter, per Human Interface Guidelines. */
    --sys-blue:   #0a84ff;
    --sys-green:  #30d158;
    --sys-indigo: #5e5ce6;
    --sys-orange: #ff9f0a;
    --sys-pink:   #ff375f;
    --sys-purple: #bf5af2;
    --sys-red:    #ff453a;
    --sys-teal:   #40c8e0;
    --sys-yellow: #ffd60a;
    --sys-gray:   #8e8e93;
    --sys-gray5:  #2c2c2e;
    --sys-gray6:  #1c1c1e;

    --accent:        #0a84ff;
    --accent-hover:  #329bff;
    --accent-soft:   rgba(10, 132, 255, 0.20);
    --green:         #4ddb85;
    --green-soft:    rgba(48, 209, 88, 0.18);
    --amber:         #ffd426;
    --amber-soft:    rgba(255, 214, 10, 0.16);
    --red:           #ff6961;
    --red-soft:      rgba(255, 69, 58, 0.18);
    --shadow-chip: 0 1px 2px rgba(0,0,0,0.5);
    --shadow-card: 0 2px 10px rgba(0,0,0,0.55);
    --shadow-lift: 0 8px 36px rgba(0,0,0,0.7);
  }}
}}

/* ===================================================================== */
/* Foundations                                                            */
/* ===================================================================== */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
  font-family: {SYSTEM_FONT};
  font-feature-settings: {INTER_FEATURES};
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-optical-sizing: auto;
  font-synthesis-weight: none;   /* use Inter's real weights, never a faux bold */
  color: var(--text);
}}

/* Inter is a variable font on the wide axis; naming the range lets the browser
   interpolate exact weights instead of snapping to the nearest static cut. */
@supports (font-variation-settings: normal) {{
  html, body, [data-testid="stAppViewContainer"] {{
    font-variation-settings: "opsz" 16;
  }}
  h1, .vr-hero__title {{ font-variation-settings: "opsz" 32; }}
  h2 {{ font-variation-settings: "opsz" 26; }}
  h3, h4 {{ font-variation-settings: "opsz" 20; }}
}}

[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1100px 620px at 12% -8%, var(--accent-soft), transparent 60%),
    linear-gradient(180deg, var(--canvas) 0%, var(--canvas-deep) 100%);
  background-attachment: fixed;
}}

code, kbd, pre, [data-testid="stCode"] {{ font-family: {MONO_FONT}; }}

/* Figures: tabular where numbers are compared, proportional in prose.
   Tabular figures share one advance width, so a column of scores stays aligned
   and a live-updating value does not jitter as its digits change. In running
   text they read badly — the gaps around a "1" are visible — so body copy keeps
   Inter's default proportional figures. Slashed zero only where a digit could be
   confused with a letter, e.g. the identifier "BLR-3/4821/2024". */
[data-testid="stMetricValue"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
.vr-locator,
.vr-pill,
.vr-row__meta,
code, pre, [data-testid="stCode"] {{
  font-variant-numeric: tabular-nums slashed-zero;
  font-feature-settings: {INTER_FEATURES}, "tnum" 1, "zero" 1;
}}

/* Typography: hierarchy from weight + size + leading as a set (§15). */
{_type_rule('h1, [data-testid="stAppViewContainer"] h1', "display")}
{_type_rule("h2", "title")}
{_type_rule("h3, h4", "heading")}
{_type_rule('p, li, [data-testid="stMarkdownContainer"] p', "body")}
{_type_rule('[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p', "caption")}

[data-testid="stCaptionContainer"] p {{ color: var(--text-dim); }}

/* Spacing in rem so layout scales with the user's text size (§15). */
.block-container {{
  padding-top: 2.6rem;
  padding-bottom: 5rem;
  max-width: 68rem;
}}

/* ===================================================================== */
/* §12 Chrome as a translucent layer, content scrolling underneath.        */
/* Bright top edge = light catching the material.                         */
/* ===================================================================== */
[data-testid="stHeader"] {{
  background: var(--chrome);
  backdrop-filter: var(--blur-chrome);
  -webkit-backdrop-filter: var(--blur-chrome);
  border-bottom: 1px solid var(--separator);
}}

/* Heavier material separates a structural region (§12). */
[data-testid="stSidebar"] {{
  background: var(--chrome);
  backdrop-filter: var(--blur-chrome);
  -webkit-backdrop-filter: var(--blur-chrome);
  border-right: 1px solid var(--separator);
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: var(--text-dim); }}
[data-testid="stSidebar"] h1 {{ font-size: 1.45rem; letter-spacing: -0.019em; }}

/* ===================================================================== */
/* Controls. Feedback on pointer-down, never on release (§1).              */
/* ===================================================================== */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {{
  font-family: {SYSTEM_FONT};
  font-size: 0.9rem;
  font-weight: 560;
  letter-spacing: -0.005em;
  border-radius: var(--r-pill);
  padding: 0.44rem 1.05rem;
  border: 1px solid var(--separator);
  background: var(--surface-solid);
  color: var(--text);
  box-shadow: var(--shadow-chip);
  transition: transform var(--dur-fast) var(--ease),
              box-shadow var(--dur-fast) var(--ease),
              opacity var(--dur-fast) var(--ease);
  will-change: transform;
}}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
  transform: translateY(-1px);
  box-shadow: var(--shadow-card);
}}
/* Pressed state is instant and reads as physical compression. */
.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] > button:active {{
  transform: scale(0.972);
  transition-duration: var(--dur-instant);
  box-shadow: var(--shadow-chip);
}}
.stButton > button:focus-visible,
[data-testid="stFormSubmitButton"] > button:focus-visible {{
  outline: none;
  box-shadow: 0 0 0 4px var(--accent-soft), var(--shadow-chip);
}}

/* One accent colour, used only for the primary path (§16 restraint). */
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {{
  background: var(--accent);
  border-color: transparent;
  color: #fff;
}}

/* ===================================================================== */
/* Cards, expanders, alerts: light material on the canvas — never light     */
/* translucency stacked on light translucency (§12).                       */
/* ===================================================================== */
[data-testid="stExpander"] {{
  border: 1px solid var(--separator);
  border-radius: var(--r-card);
  background: var(--surface);
  backdrop-filter: var(--blur-card);
  -webkit-backdrop-filter: var(--blur-card);
  box-shadow: var(--shadow-chip);
  overflow: hidden;
  transition: box-shadow var(--dur-base) var(--ease);
}}
[data-testid="stExpander"]:hover {{ box-shadow: var(--shadow-card); }}
[data-testid="stExpander"] summary {{
  font-size: 0.9rem;
  font-weight: 560;
  letter-spacing: -0.006em;
  padding: 0.72rem 1rem;
}}
[data-testid="stExpander"] summary:hover {{ color: var(--accent); }}
/* Nested surface goes solid, so translucency is never doubled. */
[data-testid="stExpander"] [data-testid="stExpander"] {{
  background: var(--surface-solid);
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}}

[data-testid="stAlert"] {{
  border-radius: var(--r-card);
  border: 1px solid var(--separator);
  box-shadow: var(--shadow-chip);
  font-size: 0.9rem;
}}

[data-testid="stMetric"] {{
  background: var(--surface);
  backdrop-filter: var(--blur-card);
  -webkit-backdrop-filter: var(--blur-card);
  border: 1px solid var(--separator);
  border-radius: var(--r-card);
  padding: 1rem 1.1rem;
  box-shadow: var(--shadow-chip);
}}
[data-testid="stMetricLabel"] p {{
  font-size: 0.75rem;
  font-weight: 560;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--text-faint);
}}
[data-testid="stMetricValue"] {{
  font-size: 1.85rem;
  font-weight: 660;
  letter-spacing: -0.021em;   /* negative tracking as type grows (§15) */
}}

/* ===================================================================== */
/* Tabs — a segmented control. Direct, specific labels (§16).              */
/* ===================================================================== */
[data-baseweb="tab-list"] {{
  gap: 0.2rem;
  padding: 0.24rem;
  border-radius: var(--r-pill);
  background: var(--surface-sunk);
  border: 1px solid var(--separator);
  backdrop-filter: var(--blur-chip);
  -webkit-backdrop-filter: var(--blur-chip);
  width: fit-content;
}}
[data-baseweb="tab-list"] button[data-baseweb="tab"] {{
  border-radius: var(--r-pill);
  padding: 0.36rem 0.95rem;
  font-size: 0.875rem;
  font-weight: 540;
  letter-spacing: -0.005em;
  color: var(--text-dim);
  background: transparent;
  border: none;
  transition: color var(--dur-fast) var(--ease),
              background var(--dur-base) var(--ease);
}}
[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {{ color: var(--text); }}
[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--text);
  background: var(--surface-solid);
  box-shadow: var(--shadow-chip);
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none; }}

/* ===================================================================== */
/* Inputs                                                                 */
/* ===================================================================== */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {{
  border-radius: var(--r-control) !important;
  border: 1px solid var(--separator) !important;
  background: var(--surface-solid) !important;
  font-size: 0.92rem;
  transition: box-shadow var(--dur-fast) var(--ease);
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
  box-shadow: 0 0 0 4px var(--accent-soft);
  border-color: var(--accent) !important;
  outline: none;
}}

[data-testid="stFileUploader"] section {{
  border: 1.5px dashed var(--separator);
  border-radius: var(--r-card);
  background: var(--surface);
  backdrop-filter: var(--blur-chip);
  -webkit-backdrop-filter: var(--blur-chip);
  transition: border-color var(--dur-base) var(--ease),
              transform var(--dur-base) var(--ease);
}}
[data-testid="stFileUploader"] section:hover {{
  border-color: var(--accent);
  transform: translateY(-1px);
}}

/* Radio options as tappable rows: proximity implies relationship (§16). */
[data-testid="stRadio"] label {{
  border-radius: var(--r-control);
  padding: 0.34rem 0.6rem;
  transition: background var(--dur-fast) var(--ease);
}}
[data-testid="stRadio"] label:hover {{ background: var(--surface-sunk); }}

/* --- iOS switch ------------------------------------------------------- */
/* The track fills with the accent and the knob translates; both animate on
   transform/background only, and the press state compresses the knob. */
[data-testid="stToggle"] [data-baseweb="checkbox"] div[role="checkbox"] {{
  background: var(--sys-gray5) !important;
  border-color: transparent !important;
  border-radius: var(--r-pill) !important;
  transition: background var(--dur-base) var(--ease) !important;
}}
[data-testid="stToggle"] [data-baseweb="checkbox"] div[role="checkbox"][aria-checked="true"] {{
  background: var(--sys-green) !important;
}}
[data-testid="stToggle"] [data-baseweb="checkbox"] div[role="checkbox"] > div {{
  background: #fff !important;
  box-shadow: 0 2px 5px rgba(0,0,0,0.22), 0 0 0 0.5px rgba(0,0,0,0.04) !important;
  transition: transform var(--dur-base) var(--ease) !important;
}}
[data-testid="stToggle"] [data-baseweb="checkbox"]:active div[role="checkbox"] > div {{
  transform: scale(0.92);
}}

/* --- iOS slider ------------------------------------------------------- */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
  background: #fff !important;
  border: 0.5px solid rgba(0,0,0,0.06) !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.22) !important;
  transition: transform var(--dur-fast) var(--ease) !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:active {{
  transform: scale(1.14);   /* grows toward the finger (§8 hint at outcome) */
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] {{ display: none; }}

/* --- progress: thin, pill, accent ------------------------------------- */
[data-testid="stProgress"] > div > div {{
  background: var(--surface-sunk) !important;
  border-radius: var(--r-pill) !important;
  height: 6px !important;
}}
[data-testid="stProgress"] > div > div > div {{
  background: linear-gradient(90deg, var(--accent), var(--sys-indigo)) !important;
  border-radius: var(--r-pill) !important;
  transition: width var(--dur-slow) var(--ease) !important;
}}

/* --- spinner: continuous, calm --------------------------------------- */
[data-testid="stSpinner"] > div {{
  border-top-color: var(--accent) !important;
  border-right-color: var(--accent-soft) !important;
  border-bottom-color: var(--accent-soft) !important;
  border-left-color: var(--accent-soft) !important;
}}
[data-testid="stSpinner"] p {{ color: var(--text-dim); font-size: 0.86rem; }}

/* --- sidebar list rows (macOS source list) ---------------------------- */
/* Selected row is a filled pill anchored to the row, matching how macOS
   sidebars indicate the current item. */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
  display: flex;
  align-items: center;
  padding: 0.42rem 0.62rem;
  border-radius: var(--r-control);
  font-size: 0.875rem;
  transition: background var(--dur-fast) var(--ease),
              color var(--dur-fast) var(--ease);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
  background: var(--accent);
  color: #fff;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
  color: #fff !important;
  font-weight: 560;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {{ display: none; }}

/* ===================================================================== */
/* Chat                                                                   */
/* ===================================================================== */
[data-testid="stChatMessage"] {{
  background: var(--surface);
  backdrop-filter: var(--blur-card);
  -webkit-backdrop-filter: var(--blur-card);
  border: 1px solid var(--separator);
  border-radius: var(--r-surface);
  box-shadow: var(--shadow-chip);
  padding: 1.05rem 1.2rem;
  margin-bottom: 0.9rem;
}}
[data-testid="stChatInput"] {{
  border-radius: var(--r-pill);
  border: 1px solid var(--separator);
  background: var(--chrome);
  backdrop-filter: var(--blur-chrome);
  -webkit-backdrop-filter: var(--blur-chrome);
  box-shadow: var(--shadow-lift);
}}

/* ===================================================================== */
/* Tables & images                                                        */
/* ===================================================================== */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border-radius: var(--r-card);
  overflow: hidden;
  border: 1px solid var(--separator);
  box-shadow: var(--shadow-chip);
}}
[data-testid="stImage"] img {{
  border-radius: var(--r-card);
  box-shadow: var(--shadow-card);
}}

/* Separators, not heavy rules: hierarchy by spacing and contrast (§16). */
hr, [data-testid="stDivider"] {{
  border-color: var(--separator);
  opacity: 1;
}}

/* §12 asks for a scroll edge effect rather than a hard 1px divider under
   floating chrome. The translucent, blurred header above already is that
   effect — content passes under it and softens. A separate sticky gradient
   would add real layout space for no visual gain, so the divider is simply
   kept hairline-faint instead. */
[data-testid="stHeader"] {{ border-bottom-color: var(--separator); }}

::-webkit-scrollbar {{ width: 11px; height: 11px; }}
::-webkit-scrollbar-thumb {{
  background: rgba(140, 140, 150, 0.4);
  border-radius: var(--r-pill);
  border: 3px solid transparent;
  background-clip: content-box;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(140, 140, 150, 0.62); background-clip: content-box; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ===================================================================== */
/* Custom components                                                      */
/* ===================================================================== */

/* Vibrancy: over translucent material, text needs more contrast and a
   slight tracking bump — not flat grey (§12). */
.vr-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.72rem;
  border-radius: var(--r-pill);
  font-size: 0.735rem;
  font-weight: 620;
  letter-spacing: 0.014em;
  border: 1px solid var(--separator);
  box-shadow: var(--shadow-chip);
  white-space: nowrap;
}}
.vr-pill__dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor;
  flex: none;
}}
.vr-pill--high   {{ color: var(--green); background: var(--green-soft); }}
.vr-pill--medium {{ color: var(--amber); background: var(--amber-soft); }}
.vr-pill--low    {{ color: var(--red);   background: var(--red-soft); }}
.vr-pill--refused{{ color: var(--text-dim); background: var(--surface-sunk); }}
.vr-pill__sep {{ opacity: 0.42; font-weight: 400; }}

/* Quoted evidence: a solid layer, so the colour is not on translucency. */
.vr-quote {{
  border-left: 3px solid var(--amber);
  border-radius: 0 var(--r-control) var(--r-control) 0;
  background: var(--surface-solid);
  padding: 0.72rem 0.95rem;
  font-size: 0.9rem;
  line-height: 1.6;
  color: var(--text);
  box-shadow: var(--shadow-chip);
}}
.vr-quote--proven {{ border-left-color: var(--green); }}

.vr-hero {{ margin-bottom: 0.4rem; }}
.vr-hero__title {{
  font-size: clamp(1.9rem, 3.4vw, 2.6rem);
  line-height: 1.06;
  letter-spacing: -0.023em;
  font-weight: 700;
  margin: 0;
}}
.vr-hero__sub {{
  font-size: 1rem;
  line-height: 1.5;
  letter-spacing: -0.004em;
  color: var(--text-dim);
  margin: 0.45rem 0 0;
  max-width: 46rem;
}}

.vr-locator {{
  font-family: {MONO_FONT};
  font-size: 0.74rem;
  letter-spacing: 0.01em;
  color: var(--text-dim);
  background: var(--surface-sunk);
  border-radius: 6px;
  padding: 0.1rem 0.4rem;
}}

/* §12 "Materialize, don't just fade": blur radius and scale animate together,
   so a glass surface reads as a real material arriving rather than an opacity
   ramp. Only transform/opacity/filter — all compositor-friendly (§11). */
@keyframes vr-materialize {{
  from {{
    opacity: 0;
    transform: translate3d(0, 8px, 0) scale(0.985);
    filter: blur(6px);
  }}
  to {{
    opacity: 1;
    transform: translate3d(0, 0, 0) scale(1);
    filter: blur(0);
  }}
}}
[data-testid="stChatMessage"], [data-testid="stMetric"] {{
  animation: vr-materialize var(--dur-base) var(--ease-out) both;
  will-change: transform, opacity;
}}

/* Staggered entrance for lists: each row arrives just after the last, which
   reads as one connected movement rather than everything popping at once. */
[data-testid="stExpander"] {{
  animation: vr-materialize var(--dur-base) var(--ease-out) both;
}}
[data-testid="stExpander"]:nth-of-type(1) {{ animation-delay: 0ms; }}
[data-testid="stExpander"]:nth-of-type(2) {{ animation-delay: 40ms; }}
[data-testid="stExpander"]:nth-of-type(3) {{ animation-delay: 80ms; }}
[data-testid="stExpander"]:nth-of-type(4) {{ animation-delay: 120ms; }}
[data-testid="stExpander"]:nth-of-type(n+5) {{ animation-delay: 160ms; }}

/* Sheet grabber — the affordance Apple puts at the top of a sheet. */
.vr-grabber {{
  width: 36px;
  height: 5px;
  border-radius: var(--r-pill);
  background: var(--sys-gray);
  opacity: 0.35;
  margin: 0 auto 0.85rem;
}}

/* Skeleton shimmer for pending content: a slow, low-contrast sweep. It stays
   above the 0.2 Hz oscillation the guidance warns about, and is removed
   entirely under reduced motion. */
@keyframes vr-shimmer {{
  from {{ background-position: -180% 0; }}
  to   {{ background-position: 180% 0; }}
}}
.vr-skeleton {{
  height: 0.85rem;
  border-radius: 6px;
  background: linear-gradient(
    90deg, var(--surface-sunk) 25%, var(--sys-gray5) 50%, var(--surface-sunk) 75%
  );
  background-size: 220% 100%;
  animation: vr-shimmer 1.4s linear infinite;
}}

/* Notification dot, for counts that should not shout. */
.vr-dot {{
  display: inline-block;
  min-width: 18px;
  padding: 0 5px;
  border-radius: var(--r-pill);
  background: var(--sys-red);
  color: #fff;
  font-size: 0.68rem;
  font-weight: 640;
  line-height: 18px;
  text-align: center;
}}

/* A row that maps a control to what it affects, with a trailing chevron —
   the familiar iOS list idiom (§16 Familiarity, grouping & mapping). */
.vr-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  padding: 0.62rem 0.9rem;
  border-radius: var(--r-control);
  background: var(--surface-solid);
  border: 1px solid var(--separator);
  box-shadow: var(--shadow-chip);
  font-size: 0.9rem;
  transition: transform var(--dur-fast) var(--ease);
}}
.vr-row:active {{ transform: scale(0.99); }}
.vr-row__meta {{ color: var(--text-dim); font-size: 0.8rem; }}
.vr-row__chevron {{ color: var(--text-faint); flex: none; }}

/* ===================================================================== */
/* §14 Accessibility. Reduced motion means gentler, not absent.           */
/* ===================================================================== */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
  }}
  /* Keep a short cross-fade; drop translation, blur and any overshoot. */
  [data-testid="stChatMessage"], [data-testid="stMetric"], [data-testid="stExpander"] {{
    animation: none !important;
    transition: opacity var(--dur-base) ease;
    filter: none;
  }}
  .stButton > button:hover,
  [data-testid="stFileUploader"] section:hover {{ transform: none; }}
  .stButton > button:active {{ transform: none; opacity: 0.72; }}
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:active {{ transform: none; }}
  .vr-row:active {{ transform: none; }}
  /* No looping motion at all. */
  .vr-skeleton {{ animation: none; background: var(--surface-sunk); }}
}}

/* Frostier and solid instead of translucent. */
@media (prefers-reduced-transparency: reduce) {{
  :root {{
    --surface: var(--surface-solid);
    --chrome:  var(--surface-solid);
    --blur-chip: none; --blur-card: none; --blur-chrome: none;
  }}
  [data-testid="stSidebar"], [data-testid="stHeader"],
  [data-testid="stChatMessage"], [data-testid="stExpander"],
  [data-testid="stMetric"], [data-testid="stChatInput"] {{
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
  }}
}}

/* Near-solid backgrounds with a defined, contrasting border. */
@media (prefers-contrast: more) {{
  :root {{
    --separator: rgba(0, 0, 0, 0.55);
    --text-dim:  #3a3a3c;
    --surface:   var(--surface-solid);
  }}
  .stButton > button, [data-testid="stExpander"], [data-testid="stMetric"] {{
    border-width: 1.5px;
  }}
}}
"""


# ---------------------------------------------------------------------------
# component helpers
# ---------------------------------------------------------------------------
_BAND_LABEL = {
    "high": "High confidence",
    "medium": "Medium confidence",
    "low": "Low confidence",
    "refused": "No evidence found",
}


def confidence_pill(band: str, groundedness: float, retrieval_score: float) -> str:
    """A status pill. Specific wording beats a bare colour (§16 feedback)."""
    key = band if band in _BAND_LABEL else "low"
    label = _BAND_LABEL[key]
    if key == "refused":
        return (
            f'<span class="vr-pill vr-pill--refused">'
            f'<span class="vr-pill__dot"></span>{label}</span>'
        )
    return (
        f'<span class="vr-pill vr-pill--{key}">'
        f'<span class="vr-pill__dot"></span>{label}'
        f'<span class="vr-pill__sep">·</span>grounded {groundedness:.0%}'
        f'<span class="vr-pill__sep">·</span>evidence {retrieval_score:.2f}'
        f"</span>"
    )


def quote_block(text: str, *, proven: bool = False, limit: int = 1400) -> str:
    """Quoted source evidence on a solid layer, never on translucency (§12).

    The text comes from a user-supplied PDF, so it is escaped (a document
    containing ``<img onerror=...>`` must not execute) and its whitespace is
    collapsed — a newline inside an injected HTML fragment would let Streamlit's
    markdown parser close the surrounding block early.
    """
    body = " ".join((text or "").split())[:limit]
    escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    modifier = " vr-quote--proven" if proven else ""
    return f'<div class="vr-quote{modifier}">{escaped}</div>'


def hero(title: str, subtitle: str) -> str:
    """Page header: tight leading and negative tracking at display size (§15)."""
    return (
        f'<div class="vr-hero"><h1 class="vr-hero__title">{title}</h1>'
        f'<p class="vr-hero__sub">{subtitle}</p></div>'
    )


def locator_chip(text: str) -> str:
    return f'<span class="vr-locator">{text}</span>'


def grabber() -> str:
    """The sheet grabber Apple puts at the top of a presented surface."""
    return '<div class="vr-grabber"></div>'


def skeleton(rows: int = 3) -> str:
    """Low-contrast placeholder while content is pending."""
    bars = "".join(
        f'<div class="vr-skeleton" style="width:{width}%;margin-bottom:0.5rem;"></div>'
        for width in (96, 88, 64)[:rows]
    )
    return bars


def list_row(label: str, meta: str = "", *, chevron: bool = True) -> str:
    """An iOS-style list row: label, trailing detail, optional chevron."""
    trailing = f'<span class="vr-row__meta">{meta}</span>' if meta else ""
    arrow = '<span class="vr-row__chevron">\u203a</span>' if chevron else ""
    return (
        f'<div class="vr-row"><span>{label}</span>'
        f'<span style="display:flex;align-items:center;gap:0.55rem;">{trailing}{arrow}</span></div>'
    )


def badge(count: int) -> str:
    """A count badge, for things that should be noticed but not shouted."""
    return f'<span class="vr-dot">{count}</span>'


def inject(st_module) -> None:
    """Apply the fonts and stylesheet. Call once, before any other output."""
    st_module.markdown(font_links() + build_css(), unsafe_allow_html=True)
