"""Theme tests — the design rules that must not silently regress.

A stylesheet is easy to break invisibly, so these assert the *invariants* the
Apple design guidance actually requires rather than exact pixel values: that
accessibility preferences are honoured, that tracking is size-specific, that
translucency is never stacked, and that quoted source text is escaped.
"""

from __future__ import annotations

import re

import pytest

from verirag.ui import theme


@pytest.fixture(scope="module")
def css() -> str:
    return theme.build_css()


class TestAccessibility:
    """§14 — reduced motion, transparency and contrast are all honoured."""

    @pytest.mark.parametrize(
        "query",
        [
            "prefers-reduced-motion: reduce",
            "prefers-reduced-transparency: reduce",
            "prefers-contrast: more",
            "prefers-color-scheme: dark",
        ],
    )
    def test_media_query_present(self, css: str, query: str):
        assert f"@media ({query})" in css

    def test_reduced_motion_keeps_a_crossfade_rather_than_nothing(self, css: str):
        block = css.split("@media (prefers-reduced-motion: reduce)")[1][:1200]
        assert "opacity" in block, "reduced motion must degrade to a cross-fade, not to nothing"

    def test_reduced_motion_stops_looping_animation(self, css: str):
        block = css.split("@media (prefers-reduced-motion: reduce)")[1][:1200]
        assert "vr-skeleton" in block, "the looping shimmer must be disabled"

    def test_reduced_transparency_drops_blur(self, css: str):
        block = css.split("@media (prefers-reduced-transparency: reduce)")[1][:900]
        assert "backdrop-filter: none" in block


class TestTypography:
    """§15 — tracking is size-specific; leading varies inversely with size."""

    def test_display_uses_negative_tracking(self):
        _size, _leading, tracking, _weight = theme.TYPE_SCALE["display"]
        assert tracking.startswith("-"), "large text needs negative tracking"

    def test_caption_uses_positive_tracking(self):
        _size, _leading, tracking, _weight = theme.TYPE_SCALE["caption"]
        assert not tracking.startswith("-") and tracking != "0"

    def test_body_tracking_is_neutral(self):
        assert theme.TYPE_SCALE["body"][2] == "0"

    def test_tracking_is_not_one_value_for_every_size(self):
        values = {entry[2] for entry in theme.TYPE_SCALE.values()}
        assert len(values) > 1, "a single letter-spacing is wrong somewhere"

    def test_leading_is_tighter_on_display_than_body(self):
        display = float(theme.TYPE_SCALE["display"][1])
        body = float(theme.TYPE_SCALE["body"][1])
        assert display < body

    def test_system_font_comes_first(self, css: str):
        assert "-apple-system" in css
        assert css.index("-apple-system") < css.index("sans-serif")


class TestInter:
    """Inter is the interface face; the platform stack is the fallback."""

    def test_inter_leads_the_stack(self):
        assert theme.SYSTEM_FONT.startswith('"Inter"')

    def test_system_fallback_is_retained_for_offline_use(self):
        """The app is meant to run without a network, so text must still render."""
        assert "-apple-system" in theme.SYSTEM_FONT
        assert theme.SYSTEM_FONT.rstrip().endswith("sans-serif")

    def test_font_links_preconnect_before_stylesheet(self):
        markup = theme.font_links()
        assert markup.index("preconnect") < markup.index('rel="stylesheet"')
        assert "fonts.gstatic.com" in markup

    def test_webfont_uses_display_swap(self):
        assert "display=swap" in theme.GOOGLE_FONTS_HREF

    def test_variable_weight_axis_requested(self):
        assert "100..900" in theme.GOOGLE_FONTS_HREF

    def test_crossorigin_on_gstatic_preconnect(self):
        assert 'href="https://fonts.gstatic.com" crossorigin' in theme.font_links()

    def test_inject_emits_fonts_then_css(self):
        captured: list[str] = []

        class FakeStreamlit:
            @staticmethod
            def markdown(body: str, unsafe_allow_html: bool = False) -> None:  # noqa: ARG004
                captured.append(body)

        theme.inject(FakeStreamlit)
        assert len(captured) == 1
        assert captured[0].index("<link") < captured[0].index("<style>")

    def test_character_variants_disambiguate_letters_from_digits(self, css: str):
        """"L37" must not read as "137" — cv05/cv08 exist for exactly that."""
        assert '"cv05" 1' in css
        assert '"cv08" 1' in css

    def test_faux_bold_is_disabled(self, css: str):
        assert "font-synthesis-weight: none" in css

    def test_optical_size_axis_varies_with_type_size(self, css: str):
        assert '"opsz" 32' in css and '"opsz" 16' in css

    def test_tabular_figures_only_where_numbers_are_compared(self, css: str):
        """Tabular figures belong in metrics and tables, never in running prose."""
        assert "tabular-nums" in css
        # Selectors of the rule that switches figures to tabular.
        selectors = css.split("font-variant-numeric: tabular-nums")[0].rsplit("*/", 1)[-1]
        assert '[data-testid="stMetricValue"]' in selectors
        assert '[data-testid="stDataFrame"]' in selectors
        for prose_selector in ('stMarkdownContainer"] p', "p, li,"):
            assert prose_selector not in selectors, "body copy must keep proportional figures"

    def test_body_keeps_neutral_tracking_despite_inter(self):
        assert theme.TYPE_SCALE["body"][2] == "0"


class TestMaterialsAndDepth:
    """§12 — translucent chrome, and never light translucency on translucency."""

    def test_chrome_is_translucent(self, css: str):
        assert "--blur-chrome" in css
        assert "backdrop-filter: var(--blur-chrome)" in css

    def test_blur_scales_with_surface_size(self, css: str):
        def radius(token: str) -> int:
            match = re.search(rf"{token}:\s*blur\((\d+)px\)", css)
            assert match, f"{token} not found"
            return int(match.group(1))

        assert radius("--blur-chip") < radius("--blur-card") < radius("--blur-chrome")

    def test_nested_surface_turns_solid(self, css: str):
        nested = '[data-testid="stExpander"] [data-testid="stExpander"]'
        assert nested in css
        block = css.split(nested)[1][:200]
        assert "backdrop-filter: none" in block, "translucency must not be stacked"

    def test_webkit_prefix_accompanies_backdrop_filter(self, css: str):
        assert css.count("-webkit-backdrop-filter") >= 5


class TestMotion:
    """§1, §4, §7, §11 — instant press feedback, no overshoot, mirrored easing."""

    def test_press_feedback_exists_and_is_fast(self, css: str):
        assert ".stButton > button:active" in css
        block = css.split(".stButton > button:active")[1][:260]
        assert "scale(" in block
        assert "var(--dur-instant)" in block

    def test_default_easing_has_no_overshoot(self):
        # A critically damped curve keeps both control-point y values within
        # [0, 1]; a value outside that range is what produces bounce.
        numbers = [float(v) for v in re.findall(r"-?\d*\.?\d+", theme.MOTION.ease)]
        assert all(0.0 <= value <= 1.0 for value in numbers[1::2])

    def test_easing_pair_is_mirrored(self):
        out = [float(v) for v in re.findall(r"-?\d*\.?\d+", theme.MOTION.ease)]
        into = [float(v) for v in re.findall(r"-?\d*\.?\d+", theme.MOTION.ease_in)]
        assert out != into, "a reversible transition needs a distinct return curve"

    def test_only_compositor_friendly_properties_animate(self, css: str):
        for keyframes in re.findall(r"@keyframes\s+vr-\w+\s*\{(.+?)\n\}", css, re.S):
            declared = set(re.findall(r"^\s*([a-z-]+)\s*:", keyframes, re.M))
            assert declared <= {"opacity", "transform", "filter", "background-position"}, declared

    def test_materialize_animates_blur_with_scale(self, css: str):
        block = css.split("@keyframes vr-materialize")[1][:400]
        assert "blur(" in block and "scale(" in block


class TestSelectorDurability:
    def test_targets_stable_attributes_not_generated_class_names(self, css: str):
        assert css.count("data-testid") > 20
        # Streamlit's hashed emotion classes look like ".css-1abc2de".
        assert not re.search(r"\.css-[0-9a-f]{6,}", css)


class TestComponents:
    def test_confidence_pill_states(self):
        for band in ("high", "medium", "low", "refused"):
            markup = theme.confidence_pill(band, 0.9, 0.5)
            assert f"vr-pill--{band}" in markup

    def test_pill_uses_words_not_only_colour(self):
        """Colour alone is not accessible feedback (§16)."""
        assert "confidence" in theme.confidence_pill("high", 0.9, 0.5).lower()

    def test_unknown_band_falls_back(self):
        assert "vr-pill--low" in theme.confidence_pill("nonsense", 0.1, 0.1)

    def test_refused_pill_omits_meaningless_numbers(self):
        assert "grounded" not in theme.confidence_pill("refused", 0.0, 0.0)

    def test_quote_block_escapes_source_text(self):
        """Quotes come from user PDFs, so they must never inject markup."""
        markup = theme.quote_block('<img src=x onerror="alert(1)">')
        assert "<img" not in markup
        assert "&lt;img" in markup

    def test_quote_block_collapses_newlines(self):
        """A newline in an injected fragment would close the HTML block early."""
        markup = theme.quote_block("first line\n\nsecond line\ttabbed")
        assert "\n" not in markup
        assert "first line second line tabbed" in markup

    def test_quote_block_escapes_ampersands(self):
        assert "&amp;" in theme.quote_block("Smith & Co.")

    def test_quote_block_truncates(self):
        assert len(theme.quote_block("x" * 5000, limit=100)) < 300

    def test_proven_quote_is_styled_differently(self):
        assert "vr-quote--proven" in theme.quote_block("text", proven=True)
        assert "vr-quote--proven" not in theme.quote_block("text", proven=False)

    def test_hero_uses_display_type(self):
        markup = theme.hero("Title", "Subtitle")
        assert "vr-hero__title" in markup and "Subtitle" in markup

    def test_list_row_and_badge_and_grabber(self):
        assert "vr-row" in theme.list_row("Label", "detail")
        assert "vr-dot" in theme.badge(3)
        assert "vr-grabber" in theme.grabber()

    def test_skeleton_renders_requested_rows(self):
        assert theme.skeleton(2).count("vr-skeleton") == 2


class TestSystemPalette:
    @pytest.mark.parametrize("token", ["--sys-blue", "--sys-green", "--sys-red", "--sys-gray"])
    def test_system_colours_defined_for_both_schemes(self, css: str, token: str):
        assert css.count(f"{token}:") >= 2, "each system colour needs a light and a dark value"

    def test_dark_scheme_brightens_the_accent(self, css: str):
        light = re.search(r"--sys-blue:\s*(#[0-9a-f]{6})", css, re.I).group(1).lower()
        dark_block = css.split("@media (prefers-color-scheme: dark)")[1]
        dark = re.search(r"--sys-blue:\s*(#[0-9a-f]{6})", dark_block, re.I).group(1).lower()
        assert light != dark


class TestInjectionSafety:
    """Regression: the stylesheet must survive Streamlit's markdown parser.

    ``st.markdown`` runs CommonMark, where **an HTML block ends at the first
    blank line**. A readable stylesheet with blank lines between sections closes
    its own ``<style>`` tag early, and every rule after that point is painted onto
    the page as literal text. Four-space-indented lines compound it by becoming
    markdown code blocks. This shipped once; these tests exist so it cannot again.
    """

    def test_no_blank_lines(self, css: str):
        body = css.replace("<style>", "").replace("</style>", "")
        assert not [line for line in body.split("\n") if not line.strip()]

    def test_no_line_is_indented_enough_to_become_a_code_block(self, css: str):
        body = css.replace("<style>", "").replace("</style>", "")
        assert not [line for line in body.split("\n") if line.startswith("    ")]

    def test_comments_are_stripped_from_the_injected_output(self, css: str):
        """Comments carry the blank lines and long rules that break the parser."""
        assert "/*" not in css and "*/" not in css

    def test_source_stylesheet_keeps_its_comments(self):
        """Readability is preserved at the source; only the output is compacted."""
        assert "/*" in theme._STYLESHEET

    def test_compaction_preserves_declarations(self):
        compacted = theme._compact("a {\n\n  color: red;  /* note */\n\n}\n")
        assert compacted == "a {\ncolor: red;\n}"

    def test_compaction_is_idempotent(self, css: str):
        body = css.replace("<style>", "").replace("</style>", "")
        assert theme._compact(body) == body

    def test_style_tag_is_a_single_unbroken_block(self, css: str):
        assert css.count("<style>") == 1
        assert css.count("</style>") == 1
        assert css.index("<style>") < css.index("</style>")

    def test_font_links_contain_no_blank_lines(self):
        assert "\n" not in theme.font_links()

    def test_injected_payload_has_no_blank_lines_at_all(self):
        captured: list[str] = []

        class FakeStreamlit:
            @staticmethod
            def markdown(body: str, unsafe_allow_html: bool = False) -> None:  # noqa: ARG004
                captured.append(body)

        theme.inject(FakeStreamlit)
        payload = captured[0]
        assert not [line for line in payload.split("\n") if not line.strip()]


class TestCssIntegrity:
    def test_braces_balance(self, css: str):
        body = css.replace("<style>", "").replace("</style>", "")
        assert body.count("{") == body.count("}")

    def test_wrapped_in_a_style_tag(self, css: str):
        assert css.strip().startswith("<style>")
        assert css.strip().endswith("</style>")

    def test_no_unresolved_format_placeholders(self, css: str):
        assert "{MOTION" not in css and "{SYSTEM_FONT" not in css
