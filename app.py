import streamlit as st
import pdfplumber
import pypdfium2 as pdfium
from utils import load_jobs_data, calculate_weighted_score, calculate_topsis_ranking

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Resume Macher", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .skill-tag { display: inline-block; padding: 4px 10px; margin: 3px; border-radius: 12px; font-weight: 600; font-size: 0.75rem; }
    .matched { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .missing { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .leaderboard-card { padding: 15px; border-radius: 10px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px; border-left: 5px solid #4f46e5; }
    
    /* NEW RULE: Forces the candidate name to be a dark, readable color */
    .leaderboard-card h3 { color: #1f2937 !important; }
    
    .rank-1 { border-left: 5px solid #fbbf24; background: #fffbeb; }
    .rank-2 { border-left: 5px solid #9ca3af; background: #f3f4f6; }
    .rank-3 { border-left: 5px solid #b45309; background: #fff7ed; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ PortfolioMatch AI: ATS Leaderboard Engine")
st.markdown("Batch process resumes and rank them using the **TOPSIS algorithm** (Technique for Order of Preference by Similarity to Ideal Solution).")
st.divider()

# --- DATA LOADING ---
jobs = load_jobs_data()
if not jobs:
    st.error("❌ **Critical: Data Missing.** Ensure `data/jobs.json` is correctly formatted.")
    st.stop()

job_options = {f"{j.get('job_title')} ({j.get('experience_required')})": j for j in jobs}

# --- UI LAYOUT ---
with st.sidebar:
    st.header("⚙️ Evaluation Metrics")
    st.info("The TOPSIS Engine evaluates candidates on a multi-dimensional matrix to find the mathematically ideal hire.")
    st.divider()
    st.subheader("Benchmark Position")
    selected_label = st.selectbox("Select Role", list(job_options.keys()), label_visibility="collapsed")
    selected_job = job_options[selected_label]
    
    st.markdown("**Core Requirements:**")
    skills_html = "".join([f'<div class="skill-tag" style="background:#e5e7eb; color:#374151;">{s.upper()}</div>' for s in selected_job.get('required_skills', [])])
    st.markdown(skills_html, unsafe_allow_html=True)

st.subheader("1️⃣ Upload Candidate Pool (PDFs)")
# NEW: accept_multiple_files=True enables batch processing
uploaded_files = st.file_uploader("Upload multiple resumes to generate a ranking.", type="pdf", accept_multiple_files=True)

if st.button("🚀 Run TOPSIS Evaluation Matrix", use_container_width=True):
    if len(uploaded_files) < 2:
        st.warning("⚠️ **Notice:** TOPSIS requires at least **two** resumes to create a comparative ranking matrix.")
    else:
        with st.spinner(f"Analyzing {len(uploaded_files)} candidates... Extracting vectors and calculating TOPSIS matrix..."):
            
            candidates_data = []
            scoring_matrix = []
            
            # --- BATCH PROCESSING LOOP ---
            for file in uploaded_files:
                resume_text = ""
                
                # Triple-Layer Fault Tolerant Parser
                try:
                    file.seek(0)
                    with pdfplumber.open(file) as pdf:
                        resume_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    if not resume_text.strip(): raise ValueError
                except Exception:
                    try:
                        file.seek(0)
                        file_bytes = file.read()
                        pdf = pdfium.PdfDocument(file_bytes)
                        text_pages = [pdf[i].get_textpage().get_text_bounded() for i in range(len(pdf))]
                        resume_text = "\n".join(text_pages)
                    except Exception:
                        continue # Skip corrupted files silently in batch mode
                
                if not resume_text.strip(): continue
                
                # Extract raw sub-scores for the matrix
                final, matched, missing, sections, kw_score, sem_score = calculate_weighted_score(resume_text, selected_job)
                
                # Store data
                candidates_data.append({
                    "Name": file.name.replace(".pdf", ""),
                    "Matched Skills": matched,
                    "Missing Skills": missing,
                    "KW_Raw": kw_score,
                    "SEM_Raw": sem_score
                })
                
                # Build the mathematical matrix for TOPSIS
                scoring_matrix.append([kw_score, sem_score])

            # --- TOPSIS ALGORITHM EXECUTION ---
            if len(scoring_matrix) > 1:
                # Weights: 70% Keywords, 30% Semantic Context | Impacts: 1 (Higher is better for both)
                topsis_scores = calculate_topsis_ranking(scoring_matrix, weights=[0.7, 0.3], impacts=[1, 1])
                
                # Merge scores back into candidate data and sort by rank
                for i in range(len(candidates_data)):
                    candidates_data[i]["TOPSIS Score"] = topsis_scores[i]
                
                ranked_candidates = sorted(candidates_data, key=lambda x: x["TOPSIS Score"], reverse=True)
                
                # --- RESULTS DASHBOARD ---
                st.divider()
                st.balloons()
                st.subheader("🏆 Candidate Leaderboard")
                
                # Render Top 3 visually
                for idx, candidate in enumerate(ranked_candidates):
                    rank_class = f"rank-{idx+1}" if idx < 3 else ""
                    medal = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🔹"
                    
                    st.markdown(f"""
                        <div class="leaderboard-card {rank_class}">
                            <h3 style="margin-top:0;">{medal} #{idx+1} | {candidate['Name']}</h3>
                            <h4 style="color: #4f46e5;">TOPSIS Closeness Coefficient: {candidate['TOPSIS Score']}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"View Skill Gap Analysis for {candidate['Name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            m_html = "".join([f'<div class="skill-tag matched">{s}</div>' for s in candidate['Matched Skills']])
                            st.markdown("**✅ Verified Skills:**<br>" + (m_html if m_html else "None"), unsafe_allow_html=True)
                        with col2:
                            miss_html = "".join([f'<div class="skill-tag missing">{s}</div>' for s in candidate['Missing Skills']])
                            st.markdown("**❌ Missing Core Skills:**<br>" + (miss_html if miss_html else "None"), unsafe_allow_html=True)
            else:
                st.error("Not enough valid readable resumes were processed to run the TOPSIS algorithm.")

st.divider()
st.caption("© 2026 PortfolioMatch AI | Python 3.13 Ready | Powered by SBERT & TOPSIS Algorithm")
