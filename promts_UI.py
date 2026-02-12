import os
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import re
import json
# ----------------------------
# 1) LOAD ENV VARIABLES
# ----------------------------
load_dotenv()


# ----------------------------
# 2) PAGE CONFIG (UI SETTINGS)
# ----------------------------
st.set_page_config(
    page_title="AI Research Tool",
    page_icon="🧠",
    layout="wide"
)


# ----------------------------
# 3) TITLE + DESCRIPTION
# ----------------------------
st.title("🧠 AI Research Tool")
st.caption("Startup Analyzer + Summarizer built with Streamlit + HuggingFace")


# ----------------------------
# 4) SIDEBAR SETTINGS PANEL
# ----------------------------
st.sidebar.header("⚙️ Settings")

model_repo = st.sidebar.selectbox(
    "Select Model",
    options=[
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ],
    index=0
)

task_mode = st.sidebar.radio(
    "Choose Mode",
    ["Startup Analyzer (VC)", "Summarize", "Custom Prompt"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.6, 0.1)
max_new_tokens = st.sidebar.slider("Max New Tokens", 64, 1024, 512, 64)


# ----------------------------
# 5) CREATE MODEL (CACHED)
# ----------------------------
@st.cache_resource
def load_model(repo_id: str):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    return ChatHuggingFace(llm=llm)


model = load_model(model_repo)


# ----------------------------
# 6) PROMPT TEMPLATES
# ----------------------------
startup_prompt_template = """
You are a strict startup analyst and VC.

Analyze the startup idea below.

Score each parameter from 0 to 10.
Be realistic and unbiased.

Return ONLY valid JSON in this exact format:

{{
  "market_size": {{ "score": 0, "reason": "" }},
  "problem_severity": {{ "score": 0, "reason": "" }},
  "solution_uniqueness": {{ "score": 0, "reason": "" }},
  "competition": {{ "score": 0, "reason": "" }},
  "monetization": {{ "score": 0, "reason": "" }},
  "execution": {{ "score": 0, "reason": "" }},
  "scalability": {{ "score": 0, "reason": "" }},
  "founder_fit": {{ "score": 0, "reason": "" }}
}}

Startup Idea:
{idea}
""".strip()

summarize_prompt_template = """
Summarize the following content in a clear and structured way.
Use bullet points.
Keep it concise.

Text:
{text}
""".strip()


# ----------------------------
# 7) MAIN INPUT AREA
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Input")
    user_input = st.text_area(
        "Enter your text / idea",
        height=250,
        placeholder="Example: An app for students to ...",
    )

with col2:
    st.subheader("📌 Quick Tips")
    st.info(
        "✅ For Startup Analyzer: Write idea in detail (target users, pricing, competitors).\n\n"
        "✅ For Summarize: Paste long text.\n\n"
        "✅ For Custom Prompt: You decide."
    )


# ----------------------------
# 8) CUSTOM PROMPT (ONLY IF MODE=Custom)
# ----------------------------
custom_prompt = ""
if task_mode == "Custom Prompt":
    st.subheader("🧩 Custom Prompt")
    custom_prompt = st.text_area(
        "Write your prompt template",
        height=150,
        placeholder="You are an expert ...",
    )


# ----------------------------
# 9) ACTION BUTTONS
# ----------------------------
run_col1, run_col2, run_col3 = st.columns([1, 1, 2])

with run_col1:
    run_btn = st.button("🚀 Run", use_container_width=True)

with run_col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)


if clear_btn:
    st.session_state.clear()
    st.rerun()


# ----------------------------
# 10) HISTORY INIT
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------
# 11) RUN MODEL
# ----------------------------
if run_btn:
    if not user_input.strip():
        st.error("❌ Input cannot be empty.")
        st.stop()

    with st.spinner("Analyzing... Please wait"):
        if task_mode == "Startup Analyzer (VC)":
            final_prompt = startup_prompt_template.format(idea=user_input)

        elif task_mode == "Summarize":
            final_prompt = summarize_prompt_template.format(text=user_input)

        else:
            if not custom_prompt.strip():
                st.error("❌ Custom prompt is empty.")
                st.stop()
            final_prompt = custom_prompt.strip() + "\n\nUser input:\n" + user_input

        result = model.invoke(final_prompt)
        output_text = result.content.strip()

        # Save to history
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": task_mode,
            "input": user_input,
            "output": output_text
        })


# ----------------------------
# 12) DISPLAY OUTPUT
# ----------------------------
if st.session_state.history:
    latest = st.session_state.history[0]

    st.divider()
    st.subheader("📤 Output")

    if latest["mode"] == "Startup Analyzer (VC)":
        try:
            analysis = extract_json(latest["output"])


            # Score calculation like your backend
            success_score = (
                analysis["market_size"]["score"] * 0.20 +
                analysis["problem_severity"]["score"] * 0.15 +
                analysis["solution_uniqueness"]["score"] * 0.15 +
                analysis["competition"]["score"] * 0.10 +
                analysis["monetization"]["score"] * 0.15 +
                analysis["execution"]["score"] * 0.10 +
                analysis["scalability"]["score"] * 0.10 +
                analysis["founder_fit"]["score"] * 0.05
            ) * 10

            analysis["success_probability"] = round(success_score, 2)

            # KPI Card
            st.metric("✅ Success Probability (%)", analysis["success_probability"])

            # JSON Viewer
            st.json(analysis)

            # Download JSON
            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(analysis, indent=2),
                file_name="startup_analysis.json",
                mime="application/json"
            )

        except Exception as e:
            st.warning("⚠️ AI output is not valid JSON.")
            st.code(latest["output"])
            st.caption(f"Error: {e}")

    else:
        st.write(latest["output"])

        st.download_button(
            label="⬇️ Download Output (.txt)",
            data=latest["output"],
            file_name="output.txt",
            mime="text/plain"
        )


# ----------------------------
# 13) HISTORY VIEWER
# ----------------------------
with st.expander("🕘 History (Previous Runs)"):
    for i, item in enumerate(st.session_state.history[:10], start=1):
        st.markdown(f"### {i}. {item['mode']} — {item['time']}")
        st.markdown("**Input:**")
        st.code(item["input"])
        st.markdown("**Output:**")
        st.code(item["output"])
        st.divider()

def extract_json(text: str):
    """
    Extract the first valid JSON object from model output.
    """
    # remove markdown ```json ``` blocks if present
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in output")

    json_str = text[start:end+1]

    return json.loads(json_str)