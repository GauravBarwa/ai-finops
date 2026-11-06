"""Shared presentation settings for the Gradio application and verification servers."""

import gradio as gr

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
    radius_size="md",
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    body_text_color="*neutral_900",
    body_text_color_dark="*neutral_100",
    body_text_color_subdued="*neutral_600",
    body_text_color_subdued_dark="*neutral_400",
    block_label_text_color="*neutral_700",
    block_label_text_color_dark="*neutral_300",
    block_background_fill="white",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_200",
    block_border_color_dark="*neutral_700",
    block_shadow="none",
    block_shadow_dark="none",
    input_background_fill_dark="*neutral_950",
    checkbox_label_background_fill_dark="*neutral_900",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_background_fill_hover_dark="*primary_500",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_primary_shadow="none",
    button_primary_shadow_dark="none",
)

CSS = """
.gradio-container {
    box-sizing: border-box !important;
    width: 100% !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 32px 24px !important;
}
#finops-header { padding: 0 0 24px; border-bottom: 1px solid var(--border-color-primary); }
#finops-header h1 { font-size: 28px; font-weight: 650; letter-spacing: -0.8px; }
#finops-header p { color: var(--body-text-color-subdued); margin-top: 8px; }
.tab-nav { gap: 16px; margin-bottom: 20px; }
.tab-nav button { padding: 14px 8px; font-weight: 600; }
.finops-panel { padding: 20px !important; gap: 16px !important; }
.finops-section { margin-top: 16px; }
.finops-note { color: var(--body-text-color-subdued); font-size: 13px; line-height: 1.6; }
.results-table { border-radius: 10px !important; max-width: 100%; }
.results-table th { font-size: 12px !important; font-weight: 600 !important; }
.results-table td { vertical-align: top !important; font-size: 13px !important; }
.results-table th, .results-table td { padding: 0 !important; line-height: 1.6 !important; }
.results-table .cell-wrap { padding: 12px !important; align-items: flex-start !important; }
.results-table .text, .results-table td, .results-table th {
    font-family: var(--font) !important;
    font-variant-numeric: tabular-nums;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere;
}
.response-panel {
    padding: 24px !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 10px;
    background: var(--block-background-fill);
    min-height: 120px;
    overflow-wrap: anywhere;
    line-height: 1.75;
}
.response-panel h2 { font-size: 18px !important; margin: 20px 0 12px !important; }
.response-panel h3 { font-size: 16px !important; }
.response-panel pre { overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
.response-panel table { display: block; max-width: 100%; overflow-x: auto; }
.decision-panel { padding: 20px !important; border-left: 3px solid var(--color-accent); }
.gradio-container footer { display: none; }
@media (max-width: 640px) {
    .gradio-container { padding: 16px 12px !important; }
    .finops-panel, .response-panel { padding: 16px !important; }
    .tab-nav { gap: 4px; }
    #finops-header h1 { font-size: 24px; }
}
@media (prefers-reduced-motion: reduce) {
    .gradio-container * { animation: none !important; transition: none !important; }
}
"""

LAUNCH_OPTIONS = {"theme": THEME, "css": CSS, "footer_links": []}
