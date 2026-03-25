import streamlit as st
import pdfplumber
import io
import pypdfium2 as pdfium
from utils import load_jobs_data, calculate_weighted_score

# --- CONFIGURATION ---
st.set_page_config(page_title="PortfolioMatch AI", page_icon="🛡️", layout="wide")

# Custom CSS for Premium Dashboard Experience
st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .skill-tag { display: inline-block; padding: 6px 12px; margin: 4px; border-radius: 12px; font-weight: 700; font-size: 0.8rem; }
    .matched { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .missing { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .gauge-container { text-align: center; padding: 20px; border-radius: 12px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e5e7eb; }
    </style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.title("🛡️ PortfolioMatch AI: Contextual Resume Benchmarking")
st.markdown("---")

# --- DATA LOADING ---
jobs = load_jobs_data()
if not jobs:
    st.error("❌ **Critical: Data Missing.** Ensure `data/jobs.json` is correctly formatted.")
    st.stop()

job_options = {f"{j.get('job_title')} ({j.get('experience_required')})": j for j in jobs}

# --- SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("⚙️ Matching Parameters")
    st.info("Logic: **70% Keywords** + **30% Semantic Similarity**.")
    st.divider()
    st.markdown("### 🧬 AI Model Details")
    st.caption("Model: `multi-qa-mpnet-base-dot-v1`\nParser: `pdfplumber` + `pypdfium2` (Fault Tolerant)")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("1️⃣ Upload Professional Resume")
    uploaded_file = st.file_uploader("Must be in PDF format", type="pdf")

with col2:
    st.subheader("2️⃣ Specify Benchmark Position")
    selected_label = st.selectbox("Role to Match Against", list(job_options.keys()))
    selected_job = job_options[selected_label]
    
    with st.expander("💼 Position Snapshot", expanded=False):
        st.write(f"**Experience Required:** {selected_job.get('experience_required')}")
        st.write(f"**Summary Highlights:** {selected_job.get('job_summary')}")
        st.markdown("**Core Skillset Requirements:**")
        skills_html = "".join([f'<div class="skill-tag" style="background:#e5e7eb; color:#374151;">{s.upper()}</div>' for s in selected_job.get('required_skills', [])])
        st.markdown(skills_html, unsafe_allow_html=True)

# --- ANALYTICS ENGINE ---
if st.button("🚀 Execute Contextual Match Analysis", use_container_width=True):
    try:
        if uploaded_file:
            with st.spinner("AI is parsing doc hierarchy and extracting technical vectors..."):
                
                resume_text = ""
                
                # --- FAULT-TOLERANT EXTRACTION ENGINE ---
                try:
                    # Primary Attempt: pdfplumber
                    uploaded_file.seek(0)
                    with pdfplumber.open(uploaded_file) as pdf:
                        resume_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    
                    if not resume_text.strip():
                        raise ValueError("Empty extraction from pdfplumber")
                        
                except Exception:
                    # Fallback Attempt: pypdfium2 (Bypasses EOF and Root object errors)
                    try:
                        uploaded_file.seek(0)
                        # Read raw bytes into memory to prevent file stream closures
                        file_bytes = uploaded_file.read()
                        pdf = pdfium.PdfDocument(file_bytes)
                        
                        text_pages = []
                        for i in range(len(pdf)):
                            page = pdf[i]
                            textpage = page.get_textpage()
                            text_pages.append(textpage.get_text_bounded())
                            
                        resume_text = "\n".join(text_pages)
                    except Exception as final_e:
                        st.error(f"❌ **Document Corruption is too severe**: {str(final_e)}")
                        st.stop()

                # Final Validation
                if not resume_text.strip():
                    st.error("❌ **Failure: Document Unreadable.** The PDF may be an image scan requiring OCR.")
                    st.stop()
                
                # Execute Scoring Logic from utils.py
                score, matched, missing, sections = calculate_weighted_score(resume_text, selected_job)
                
                # UI Results Rendering
                st.divider()
                st.balloons()
                
                tab_res, tab_skills, tab_sections = st.tabs(["📊 Matching Dashboard", "🔍 Skill Gap Profile", "📑 Document Segment Analysis"])
                
                with tab_res:
                    st.markdown(f"""
                        <div class="gauge-container">
                            <h3>Overall Match Quality</h3>
                            <h2 style="color: #4f46e5; font-size: 3.5rem; margin: 10px 0;">{score}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(score / 100)
                    
                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        st.subheader("🏁 Readiness Assessment")
                        if score >= 80:
                            st.success("🌟 **High-Priority Hire Profile**: Strong technical alignment.")
                        elif score >= 50:
                            st.warning("⚖️ **Balanced Profile**: Covers foundational requirements.")
                        else:
                            st.error("⚠️ **Low Profile Alignment**: Significant gaps detected.")
                    
                    with r_col2:
                        st.subheader("💡 Strategic Advice")
                        if missing:
                            st.info(f"Adding **{missing[0]}** to your profile could significantly improve this ranking.")
                        else:
                            st.success("Your resume perfectly reflects the requirements of this role.")

                with tab_skills:
                    s_col1, s_col2 = st.columns(2)
                    with s_col1:
                        st.markdown(f"✅ **Validated Technical Skills ({len(matched)})**")
                        m_html = "".join([f'<div class="skill-tag matched">{s}</div>' for s in matched])
                        st.markdown(m_html, unsafe_allow_html=True)
                    
                    with s_col2:
                        st.markdown(f"❌ **Prioritized Requirement Gaps ({len(missing)})**")
                        miss_html = "".join([f'<div class="skill-tag missing">{s}</div>' for s in missing])
                        st.markdown(miss_html, unsafe_allow_html=True)

                with tab_sections:
                    st.subheader("Segments Identified by AI Engine")
                    if sections:
                        s_cols = st.columns(len(sections))
                        for i, (name, content) in enumerate(sections.items()):
                            with s_cols[i % len(sections)]:
                                st.markdown(f"**{name.upper()}**")
                                st.caption(f"{len(content.split())} words")
                                with st.expander("Review Segment"):
                                    st.text(content[:800] + ("..." if len(content) > 800 else ""))
                    else:
                        st.info("No distinct sections identified; processing as a unified document.")
        else:
            st.warning("⚠️ Action Required: Please upload your resume as a PDF.")

    except Exception as e:
        st.error(f"❌ **System Error Encountered:** {str(e)}")
        st.info("💡 **Troubleshooting**: If this error persists, try 'Exporting as PDF' again from your source document.")

# --- FOOTER ---
st.divider()
st.caption("© 2026 PortfolioMatch AI | Python 3.13 Ready | Multi-Parser Hybrid Engine")