import streamlit as st
import pickle
import plotly.graph_objects as go
import re
import html as html_lib

model = pickle.load(open("models/toxic_model.pkl", "rb"))

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

label_display = {
    "toxic":         "Harmful Language",
    "severe_toxic":  "Extreme Content",
    "obscene":       "Explicit Content",
    "threat":        "Threatening",
    "insult":        "Personal Attack",
    "identity_hate": "Hate Speech",
}

label_description = {
    "toxic":         "General hostility, rudeness, or harmful tone",
    "severe_toxic":  "Extremely aggressive or hateful language",
    "obscene":       "Sexually explicit or grossly offensive content",
    "threat":        "Expressions of intent to harm someone",
    "insult":        "Direct personal attacks or degrading remarks",
    "identity_hate": "Hatred targeting race, religion, gender, or ethnicity",
}

# Words to visually highlight in the analyzed message
flaggable_words = [
    "idiot", "trash", "useless", "moron", "stupid", "kill", "hate",
    "pathetic", "dumb", "garbage", "worst", "loser", "clown", "retard",
    "dogshit", "fuck", "bitch", "asshole", "cunt", "kys", "nigger",
    "fag", "fucking", "die", "cancer", "scum", "filth",
]

# Severe terms that bump the score up when the ML model under-detects
slang_boost_words = [
    "fuck", "fucking", "bitch", "retard", "asshole", "cunt",
    "nigger", "fag", "kys", "dogshit", "kill yourself",
]

# Common ways people try to sneak past basic word filters
leet_map = {
    r'f[\W_*@#]*[u*][\W_*@#]*c[\W_*@#]*k': 'fuck',
    r'[a@][\W_]*[s$][\W_]*[s$]':            'ass',
    r'[s$]h[\W_]*[i!1][\W_]*t':             'shit',
    r'b[i!1]tch':                            'bitch',
    r'n[i!1]g+[e3]r':                       'nigger',
    r'\bkys\b':                              'kill yourself',
    r'\bkms\b':                              'kill myself',
    r'\bstfu\b':                             'shut the fuck up',
    r'\bwtf\b':                              'what the fuck',
}

# Phrase-level patterns for racist dog-whistles and coded language.
# The ML model misses these because they rely on multi-word context,
# not individual word frequencies.
implicit_bias_patterns = [
    (r'\byour\s+kind\b',                                                        18, "identity_hate"),
    (r'\byou\s+people\b',                                                       12, "identity_hate"),
    (r'\bpeople\s+like\s+you\b',                                                10, "identity_hate"),
    (r'\bcotton\s+(fields?|picking|gin|plantation)\b',                          25, "identity_hate"),
    (r'\bplow(ing)?\s+(the\s+)?cotton\b',                                       28, "identity_hate"),
    (r'\bpicking\s+cotton\b',                                                   28, "identity_hate"),
    (r'\bplantation\b',                                                         15, "identity_hate"),
    (r'\bback\s+to\s+(the\s+)?fields?\b',                                       18, "identity_hate"),
    (r'\bgo\s+back\s+to\s+(africa|your\s+country|where\s+you\s+came)\b',        30, "identity_hate"),
    (r'\bwhere\s+you\s+(came\s+from|belong)\b',                                 14, "identity_hate"),
    (r'\bnot\s+welcome\s+here\b',                                               12, "identity_hate"),
    (r'\bdid(n.t|\s+not)\s+know\s+your\s+kind\b',                               30, "identity_hate"),
    (r'\baren.t\s+you\s+supposed\s+to\s+be\b',                                  20, "identity_hate"),
    (r'\bsupposed\s+to\s+be\s+(in\s+the\s+(fields?|cotton)|plowing|picking)\b', 30, "identity_hate"),
    (r'\ballowed\s+to\s+be\s+(free|out|here)\b',                                20, "identity_hate"),
    (r'\b(you.re|you\s+are|acts?\s+like)\s+an?\s+animal\b',                     12, "identity_hate"),
    (r'\b(you|they).re\s+all\s+the\s+same\b',                                  14, "identity_hate"),
    (r'\b(go\s+back|return)\s+to\s+(your\s+)?(country|homeland|desert|jungle)\b', 25, "identity_hate"),
    (r'\bshould(n.t|\s+not)\s+be\s+(allowed|here|free)\b',                      16, "identity_hate"),
    (r'\bknow\s+your\s+place\b',                                                18, "insult"),
    (r'\bstay\s+in\s+your\s+(place|lane|corner)\b',                             14, "insult"),
]

# Common gaming phrases that look bad in isolation but are just normal banter.
# Only applies when no severe slang or bias patterns are present; so
# "git gud you fucking idiot" still scores high.
gaming_banter_phrases = [
    (r'\bgit\s+gud\b',         -18),
    (r'\bget\s+good\b',        -15),
    (r'\bnoob\b',              -15),
    (r'\bn00b\b',              -15),
    (r'\bl2p\b',               -15),
    (r'\blearn\s+to\s+play\b', -12),
    (r'\bskill\s+issue\b',     -18),
    (r'\bgg\s+ez\b',           -10),
    (r'\btry\s+hard\b',        -10),
    (r'\bbot\s+(play|player)\b', -12),
]

severity_weights = {
    "severe_toxic":  1.4,
    "threat":        1.3,
    "identity_hate": 1.2,
    "obscene":       1.1,
    "insult":        1.0,
    "toxic":         0.9,
}

detection_threshold  = 25.0
borderline_threshold = 50.0


st.set_page_config(page_title="PrimalSignal", page_icon="", layout="wide")

if "history"        not in st.session_state: st.session_state.history        = []
if "total_analyzed" not in st.session_state: st.session_state.total_analyzed = 0
if "total_flagged"  not in st.session_state: st.session_state.total_flagged  = 0

primary        = "#ea580c"
primary_dark   = "#c2410c"
primary_light  = "#fff7ed"
primary_border = "#fed7aa"
primary_text   = "#c2410c"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

.stApp {{
    background-color: #f8fafc;
    background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
    background-size: 24px 24px;
    font-family: 'Inter', -apple-system, sans-serif;
}}
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
    color: #0f172a;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 2.5rem 4rem; max-width: 1360px; }}

textarea {{
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 14px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.7 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
textarea:focus {{
    border-color: {primary} !important;
    box-shadow: 0 0 0 4px rgba(234,88,12,0.1) !important;
    outline: none !important;
}}
textarea::placeholder {{ color: #cbd5e1 !important; }}
.stTextArea label {{ display: none !important; }}

/* Main analyze button — primary orange */
.stButton > button {{
    background: {primary} !important;
    color: white !important;
    border-radius: 12px !important;
    height: 50px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 1px 2px rgba(234,88,12,0.2), 0 4px 12px rgba(234,88,12,0.15) !important;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s !important;
}}
.stButton > button:hover {{
    background: {primary_dark} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 4px rgba(234,88,12,0.3), 0 8px 20px rgba(234,88,12,0.2) !important;
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}

[data-testid="stProgressBar"] > div > div {{
    background: #f1f5f9 !important;
    border-radius: 99px !important;
    height: 5px !important;
}}
[data-testid="stProgressBar"] > div > div > div {{
    background: linear-gradient(90deg, {primary} 0%, #f97316 100%) !important;
    border-radius: 99px !important;
}}
[data-testid="stAlert"] {{
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    border-radius: 12px !important;
    color: #92400e !important;
    font-size: 13px !important;
}}
hr {{ border-color: #f1f5f9 !important; }}
</style>
""", unsafe_allow_html=True)


# Core Logic

def preprocess(text):
    cleaned = text.lower()
    found = []
    for pattern, replacement in leet_map.items():
        if re.search(pattern, cleaned, re.IGNORECASE):
            found.append(replacement)
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    # Normalize looooong repeated characters (loooser -> looser)
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
    return cleaned, found


def analyze(message):
    cleaned, obfuscations = preprocess(message)
    raw_probs = model.predict_proba([cleaned])[0]
    scores = {label: float(raw_probs[i]) * 100 for i, label in enumerate(label_names)}

    weighted     = {k: scores[k] * severity_weights[k] for k in label_names}
    top_category = max(weighted, key=weighted.get)
    base_score   = scores[top_category]

    boost, matched = 0, 0
    for word in slang_boost_words:
        if matched >= 3: break
        if re.search(rf'\b{re.escape(word)}\b', message.lower()):
            boost += 10
            matched += 1

    if obfuscations:
        boost += min(len(obfuscations) * 8, 20)

    bias_hits = []
    for pattern, pattern_boost, affected_label in implicit_bias_patterns:
        if re.search(pattern, message.lower(), re.IGNORECASE):
            boost += pattern_boost
            bias_hits.append(affected_label)
            scores[affected_label] = min(scores[affected_label] + pattern_boost, 100.0)

    # Apply gaming banter reduction only when no severe slang or bias detected,
    # so "git gud" gets softened but "git gud you f*cking idiot" does not.
    has_severe = matched > 0 or len(bias_hits) > 0 or len(obfuscations) > 0
    gaming_reduction = 0
    if not has_severe:
        for pattern, reduction in gaming_banter_phrases:
            if re.search(pattern, message.lower(), re.IGNORECASE):
                gaming_reduction += reduction

    final_score = max(min(base_score + boost + gaming_reduction, 100.0), 0.0)

    # Re-rank after all boosts
    weighted2    = {k: scores[k] * severity_weights[k] for k in label_names}
    top_category = max(weighted2, key=weighted2.get)

    flagged_categories = [l for l, v in scores.items() if v > 15.0]

    if final_score >= 90:   sev = "Critical"
    elif final_score >= 70: sev = "High"
    elif final_score >= 40: sev = "Medium"
    else:                   sev = "Low"

    if final_score < detection_threshold:     verdict = "safe"
    elif final_score < borderline_threshold:  verdict = "borderline"
    else:                                     verdict = "toxic"

    word_count = len(message.split())
    low_confidence = word_count < 4

    return {
        "score":          final_score,
        "top_category":   top_category,
        "scores":         scores,
        "flagged":        flagged_categories,
        "severity":       sev,
        "verdict":        verdict,
        "obfuscations":   obfuscations,
        "bias_hits":      list(set(bias_hits)),
        "char_count":     len(message),
        "word_count":     word_count,
        "low_confidence": low_confidence,
    }


def highlight_message(message, obfuscations):
    result = html_lib.escape(message)
    for word in flaggable_words:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        result  = pattern.sub(
            r'<span style="color:#dc2626;font-weight:600;background:#fef2f2;padding:1px 5px;border-radius:4px;">\1</span>',
            result,
        )
    for replacement in obfuscations:
        pattern = re.compile(rf'\b({re.escape(replacement)})\b', re.IGNORECASE)
        result  = pattern.sub(
            r'<span style="color:#d97706;font-weight:600;background:#fffbeb;padding:1px 5px;border-radius:4px;text-decoration:underline dotted;">\1 ⚠</span>',
            result,
        )
    return result


def make_chart(scores):
    cats   = [label_display[k] for k in label_names]
    values = [scores[k] for k in label_names]
    colors = [primary if v > detection_threshold else "#e2e8f0" for v in values]
    fig = go.Figure(go.Bar(
        x=cats, y=values, marker_color=colors, marker_line_width=0,
        hovertemplate="%{x}: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False,
                   range=[0, 100], ticksuffix="%", tickfont=dict(size=10, color="#94a3b8")),
        margin=dict(l=0, r=0, t=10, b=0), height=220, showlegend=False, bargap=0.4,
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                        font=dict(family="Inter, sans-serif", color="#0f172a", size=12)),
    )
    return fig


def sec(title):
    return (
        f'<p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'
        f'color:#94a3b8;margin:1.6rem 0 0.65rem;padding-bottom:0.5rem;'
        f'border-bottom:1.5px solid #f1f5f9;font-family:Inter,sans-serif;">{title}</p>'
    )

base_chip = (
    "border-radius:7px;padding:4px 11px;font-size:11px;font-weight:700;"
    "letter-spacing:0.04em;text-transform:uppercase;display:inline-block;"
)
cat_chip  = f"color:{primary_text};background:{primary_light};border:1.5px solid {primary_border};"
sev_styles = {
    "Critical": "color:#be123c;background:#fff1f2;border:1.5px solid #fecdd3;",
    "High":     "color:#c2410c;background:#fff7ed;border:1.5px solid #fed7aa;",
    "Medium":   "color:#92400e;background:#fefce8;border:1.5px solid #fde68a;",
    "Low":      "color:#065f46;background:#f0fdf4;border:1.5px solid #bbf7d0;",
}


# Page

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding-bottom:1.5rem;margin-bottom:2rem;border-bottom:1px solid #e2e8f0;">
    <div style="font-size:22px;font-weight:800;letter-spacing:-0.04em;
                color:#0f172a;font-family:Inter,sans-serif;">
        Primal<span style="color:{primary};">Signal</span>
    </div>
    <div style="font-size:12px;font-weight:500;color:#94a3b8;">Developed by Hamdan Tayyab aka PeachSpade</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="margin-bottom:1.75rem;">
    <div style="font-size:36px;font-weight:800;letter-spacing:-0.04em;line-height:1.12;
                color:#0f172a;margin-bottom:0.65rem;">Are you a toxic gamer?</div>
    <div style="font-size:15px;font-weight:400;color:#64748b;max-width:430px;line-height:1.65;">
        Type something you'd say in a game and see if it would get you flagged.
    </div>
</div>
""", unsafe_allow_html=True)

clean_count = st.session_state.total_analyzed - st.session_state.total_flagged
flag_rate   = (
    f"{(st.session_state.total_flagged / st.session_state.total_analyzed * 100):.0f}% flag rate"
    if st.session_state.total_analyzed > 0 else "No messages yet"
)
pill = (
    "background:white;border:1.5px solid #e2e8f0;border-radius:10px;"
    "padding:8px 18px;font-size:12px;font-weight:500;color:#64748b;"
    "box-shadow:0 1px 3px rgba(0,0,0,0.05);"
)
st.markdown(f"""
<div style="display:flex;gap:0.6rem;margin-bottom:1.75rem;flex-wrap:wrap;">
    <div style="{pill}">Analyzed <strong style="color:#0f172a;">{st.session_state.total_analyzed}</strong></div>
    <div style="{pill}">Flagged <strong style="color:#dc2626;">{st.session_state.total_flagged}</strong></div>
    <div style="{pill}">Clean <strong style="color:#059669;">{clean_count}</strong></div>
    <div style="{pill}"><strong style="color:#64748b;">{flag_rate}</strong></div>
</div>
""", unsafe_allow_html=True)

left_col, _, right_col = st.columns([2.1, 0.08, 1.1])

with left_col:

    user_input = st.text_area(
        "msg",
        height=155,
        placeholder="Paste or type a message to analyze…",
        label_visibility="collapsed",
    )

    char_count = len(user_input)
    char_color = "#dc2626" if char_count > 500 else "#94a3b8"
    st.markdown(
        f'<div style="text-align:right;font-size:11px;font-weight:500;color:{char_color};'
        f'margin-top:-0.25rem;margin-bottom:0.6rem;">{char_count} chars</div>',
        unsafe_allow_html=True,
    )

    analyze_btn = st.button("Analyze Message", use_container_width=True)

    if analyze_btn:
        msg = user_input.strip()
        if not msg:
            st.warning("Please enter a message to analyze.")
        else:
            result = analyze(msg)

            st.session_state.total_analyzed += 1
            if result["verdict"] != "safe":
                st.session_state.total_flagged += 1
            st.session_state.history.insert(0, {
                "message":      msg,
                "score":        result["score"],
                "verdict":      result["verdict"],
                "top_category": result["top_category"],
            })
            st.session_state.history = st.session_state.history[:20]

            # Verdict card
            if result["verdict"] == "toxic":
                card_bg, card_border = "#fff1f2", "#fecdd3"
                eye_color, score_color = "#e11d48", "#be123c"
                eye_label = "Toxic Content Detected"
            elif result["verdict"] == "borderline":
                card_bg, card_border = "#fffbeb", "#fde68a"
                eye_color, score_color = "#d97706", "#92400e"
                eye_label = "Borderline — Could Be Flagged"
            else:
                card_bg, card_border = "#f0fdf4", "#bbf7d0"
                eye_color, score_color = "#059669", "#065f46"
                eye_label = "Message Looks Clean"

            desc_text = (
                f"Risk score · Primary: {label_display[result['top_category']]}"
                if result["verdict"] != "safe"
                else f"Risk score · Below the {detection_threshold:.0f}% detection threshold"
            )

            st.markdown(f"""
<div style="background:{card_bg};border:1.5px solid {card_border};border-radius:18px;
            padding:1.75rem 2rem 1.5rem;margin:1.25rem 0;
            box-shadow:0 2px 8px rgba(0,0,0,0.04),0 8px 24px rgba(0,0,0,0.03);">
    <p style="font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
              color:{eye_color};margin:0 0 0.4rem;font-family:Inter,sans-serif;">{eye_label}</p>
    <p style="font-size:58px;font-weight:900;letter-spacing:-0.06em;line-height:1;
              color:{score_color};margin:0 0 0.3rem;font-family:Inter,sans-serif;">
        {result['score']:.0f}<span style="font-size:28px;font-weight:600;opacity:0.6;">%</span>
    </p>
    <p style="font-size:13px;color:#64748b;margin:0 0 1.1rem;font-family:Inter,sans-serif;">{desc_text}</p>
</div>""", unsafe_allow_html=True)

            # Chips row
            chips  = f'<span style="{base_chip}{sev_styles[result["severity"]]}">{result["severity"]}</span>'
            chips += f'&nbsp;<span style="{base_chip}{cat_chip}">{label_display[result["top_category"]]}</span>'
            for lbl in result["flagged"]:
                if lbl != result["top_category"]:
                    chips += f'&nbsp;<span style="{base_chip}{cat_chip}">{label_display[lbl]}</span>'
            if result["obfuscations"]:
                chips += f'&nbsp;<span style="{base_chip}color:#d97706;background:#fffbeb;border:1.5px solid #fde68a;">Obfuscation Detected</span>'
            if result["bias_hits"]:
                chips += f'&nbsp;<span style="{base_chip}color:#7c2d12;background:#fff7ed;border:1.5px solid #fdba74;">Implicit Bias Detected</span>'
            if result["low_confidence"]:
                chips += f'&nbsp;<span style="{base_chip}color:#475569;background:#f8fafc;border:1.5px solid #e2e8f0;">Short Message</span>'

            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.25rem;">{chips}</div>', unsafe_allow_html=True)

            st.markdown(
                f'<p style="font-size:11px;color:#cbd5e1;margin-top:0.4rem;">'
                f'{result["word_count"]} words · {result["char_count"]} characters</p>',
                unsafe_allow_html=True,
            )

            # Short message notice
            if result["low_confidence"]:
                st.markdown(
                    f'<div style="margin-top:0.75rem;background:#f8fafc;border:1.5px solid #e2e8f0;'
                    f'border-radius:10px;padding:0.7rem 1rem;font-size:13px;color:#64748b;line-height:1.55;">'
                    f'Short messages are harder to analyze accurately. More context improves the result.</div>',
                    unsafe_allow_html=True,
                )

            # Flagged terms
            st.markdown(sec("Flagged Terms"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:white;border:1.5px solid #f1f5f9;border-radius:14px;'
                f'padding:1rem 1.25rem;font-size:15px;line-height:1.75;color:#334155;'
                f'box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
                f'{highlight_message(msg, result["obfuscations"])}</div>',
                unsafe_allow_html=True,
            )
            if result["obfuscations"]:
                decoded = ", ".join(set(result["obfuscations"]))
                st.markdown(
                    f'<p style="margin-top:0.5rem;font-size:12px;color:#d97706;font-weight:500;">'
                    f'Decoded obfuscation: {html_lib.escape(decoded)}</p>',
                    unsafe_allow_html=True,
                )
            if result["bias_hits"]:
                st.markdown(
                    f'<div style="margin-top:0.75rem;background:#fff7ed;border:1.5px solid #fed7aa;'
                    f'border-radius:10px;padding:0.7rem 1rem;font-size:13px;color:#92400e;line-height:1.55;">'
                    f'<strong>Implicit bias detected.</strong> This message uses phrasing patterns '
                    f'associated with racist dog-whistles — the individual words may look harmless, '
                    f'but the combination signals something else.</div>',
                    unsafe_allow_html=True,
                )

            # Chart
            st.markdown(sec("Category Breakdown"), unsafe_allow_html=True)
            st.plotly_chart(make_chart(result["scores"]), use_container_width=True, config={"displayModeBar": False})

            # Score rows with descriptions
            st.markdown(sec("Confidence Scores"), unsafe_allow_html=True)
            for label in label_names:
                val  = result["scores"][label]
                desc = label_description[label]
                dot  = (
                    f'<span style="font-size:10px;font-weight:700;color:{primary};margin-left:5px;">●</span>'
                    if val > detection_threshold else ""
                )
                st.markdown(
                    f'<div style="margin-bottom:4px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:13px;font-weight:600;color:#334155;">{label_display[label]}{dot}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#0f172a;">{val:.1f}%</span>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(int(min(val, 100)))


with right_col:

    history = st.session_state.history

    count_html = (
        f'<span style="background:#f1f5f9;color:#64748b;font-size:11px;font-weight:700;'
        f'padding:3px 10px;border-radius:99px;">{len(history)}</span>'
        if history else ""
    )
    st.markdown(
        f'<div style="background:white;border:1.5px solid #e2e8f0;border-radius:16px 16px 0 0;'
        f'padding:0.9rem 1.1rem;display:flex;align-items:center;justify-content:space-between;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.04);">'
        f'<span style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
        f'color:#94a3b8;font-family:Inter,sans-serif;">Moderation Log</span>'
        f'{count_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not history:
        st.markdown(
            f'<div style="background:white;border:1.5px solid #e2e8f0;border-top:none;'
            f'border-radius:0 0 16px 16px;padding:2.75rem 1.5rem;text-align:center;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.03);">'
            f'<p style="font-size:13px;color:#cbd5e1;margin:0;font-weight:500;">No messages yet</p>'
            f'<p style="font-size:12px;color:#e2e8f0;margin:0.3rem 0 0;">Results appear as you analyze</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        entries = history[:12]
        for i, entry in enumerate(entries):
            dot_c   = "#ef4444" if entry["verdict"] != "safe" else "#10b981"
            score_c = "#dc2626" if entry["verdict"] == "toxic" else ("#d97706" if entry["verdict"] == "borderline" else "#059669")
            preview = html_lib.escape(entry["message"])
            if len(preview) > 38:
                preview = preview[:38] + "…"
            cat_lbl = label_display.get(entry["top_category"], "")
            br      = "0 0 16px 16px" if i == len(entries) - 1 else "0"
            st.markdown(
                f'<div style="background:white;border:1.5px solid #e2e8f0;border-top:none;'
                f'border-radius:{br};padding:0.65rem 1.1rem;display:flex;align-items:center;gap:0.75rem;">'
                f'<div style="width:7px;height:7px;min-width:7px;border-radius:50%;'
                f'background:{dot_c};box-shadow:0 0 6px {dot_c}55;flex-shrink:0;"></div>'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:12px;color:#64748b;overflow:hidden;text-overflow:ellipsis;'
                f'white-space:nowrap;font-style:italic;">"{preview}"</div>'
                f'<div style="font-size:10px;color:#cbd5e1;margin-top:1px;">{cat_lbl}</div>'
                f'</div>'
                f'<div style="font-size:12px;font-weight:700;color:{score_c};flex-shrink:0;">{entry["score"]:.0f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Session summary row at the bottom of the log
        if st.session_state.total_analyzed > 0:
            rate = st.session_state.total_flagged / st.session_state.total_analyzed * 100
            bar_w = f"{rate:.0f}%"
            st.markdown(
                f'<div style="background:#fafafa;border:1.5px solid #e2e8f0;border-top:none;'
                f'border-radius:0 0 16px 16px;padding:0.75rem 1.1rem;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                f'<span style="font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;">Session Flag Rate</span>'
                f'<span style="font-size:11px;font-weight:700;color:#0f172a;">{rate:.0f}%</span>'
                f'</div>'
                f'<div style="background:#e2e8f0;border-radius:99px;height:4px;">'
                f'<div style="background:{primary};width:{bar_w};height:4px;border-radius:99px;'
                f'transition:width 0.4s ease;max-width:100%;"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if history:
        if st.button("Clear Log", use_container_width=True):
            st.session_state.history        = []
            st.session_state.total_analyzed = 0
            st.session_state.total_flagged  = 0
            st.rerun()
