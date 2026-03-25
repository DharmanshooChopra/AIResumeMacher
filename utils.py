import json
import re
import spacy
import os
from sentence_transformers import SentenceTransformer, util

# --- ROBUST MODEL LOADING ---
def load_spacy_model():
    """Ensures en_core_web_sm is downloaded and loaded."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        try:
            return spacy.load("en_core_web_sm")
        except:
            return spacy.blank("en")

nlp = load_spacy_model()

# Load high-performance model for long-short text matching
# 'multi-qa-mpnet-base-dot-v1' is superior for dot-product semantic search
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# --- CONFIGURATION & WEIGHTS ---
# Section weights for hierarchy-based analysis
SECTION_WEIGHTS = {
    "skills": 1.0, 
    "experience": 0.8, 
    "projects": 0.7, 
    "education": 0.4, 
    "others": 0.2
}

# Pre-compiled Regex patterns for Python 3.13+ compatibility (Avoiding 'global flags' error)
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
PHONE_PATTERN = re.compile(r'\+?\d[\d -]{8,}\d')
WHITESPACE_PATTERN = re.compile(r'\s+')

SECTION_HEADERS = {
    "skills": re.compile(r'(skills|technologies|technical skillset|competencies|expertise|proficiencies|tools|technical strengths|stack)\b', re.IGNORECASE),
    "experience": re.compile(r'(experience|employment|work history|professional history|career summary|professional experience|internships|work experience)\b', re.IGNORECASE),
    "projects": re.compile(r'(projects|academic projects|personal projects|key projects|academic achievements|portfolios)\b', re.IGNORECASE),
    "education": re.compile(r'(education|academic background|qualifications|academic qualifications|degrees|certifications)\b', re.IGNORECASE),
}

def load_jobs_data():
    """Loads job definitions from the structured JSON file."""
    try:
        with open("data/jobs.json", "r") as f:
            data = json.load(f)
        return data["jobs"]
    except Exception:
        return []

def clean_text(text):
    """Refined text cleaning using re.compile for Python 3.13+ safety."""
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = PHONE_PATTERN.sub('', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text

def extract_sections(text):
    """
    Segmentation engine using multiline-safe regex compilation.
    Identifies high-priority and low-priority regions of the resume.
    """
    text = text.replace('\r\n', '\n')
    lines = text.split('\n')
    
    section_map = []
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: continue
        
        for name, pattern in SECTION_HEADERS.items():
            if pattern.match(clean_line):
                section_map.append({'name': name, 'line_idx': i})
                break
    
    parsed_sections = { "skills": "", "experience": "", "projects": "", "education": "", "others": "" }
    
    if not section_map:
        parsed_sections["others"] = text
        return parsed_sections
    
    # Text before the first header goes to 'others'
    first_header_line = section_map[0]['line_idx']
    parsed_sections["others"] = "\n".join(lines[:first_header_line])
    
    for i in range(len(section_map)):
        start_line = section_map[i]['line_idx']
        end_line = section_map[i+1]['line_idx'] if i+1 < len(section_map) else len(lines)
        
        name = section_map[i]['name']
        content = "\n".join(lines[start_line:end_line]).strip()
        
        if parsed_sections[name]:
            parsed_sections[name] += "\n\n" + content
        else:
            parsed_sections[name] = content
            
    return {k: v for k, v in parsed_sections.items() if v}

def calculate_weighted_score(resume_text, job_obj):
    """
    Production-ready Scoring Engine:
    - 70% Keyword Accuracy (Weighted by Section Hierarchy)
    - 30% Semantic Similarity (Dot-product SBERT)
    """
    required_skills = [s.lower() for s in job_obj.get('required_skills', [])]
    job_summary = job_obj.get('job_summary', "")
    sections = extract_sections(resume_text)
    
    # --- 1. KEYWORD MATCHING (70%) ---
    total_found_weight = 0
    matched_skills = []
    missing_skills = []
    
    resume_lower = resume_text.lower()
    
    for skill in required_skills:
        skill_best_weight = 0
        is_found = False
        
        # Check specific sections for weighted credit
        for sec_name, sec_content in sections.items():
            if skill in sec_content.lower():
                is_found = True
                skill_best_weight = max(skill_best_weight, SECTION_WEIGHTS.get(sec_name, 0.2))
        
        # Backup check for entire text if not found in specific sections
        if not is_found and skill in resume_lower:
            is_found = True
            skill_best_weight = 0.2
            
        if is_found:
            total_found_weight += skill_best_weight
            matched_skills.append(skill.upper())
        else:
            missing_skills.append(skill.upper())
            
    keyword_score = (total_found_weight / len(required_skills)) * 100 if required_skills else 0
    
    # --- 2. SEMANTIC SIMILARITY (30%) ---
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(f"{job_summary} {' '.join(required_skills)}")
    
    resume_emb = model.encode(clean_resume, convert_to_tensor=True)
    jd_emb = model.encode(clean_jd, convert_to_tensor=True)
    
    semantic_score = float(util.cos_sim(resume_emb, jd_emb)[0][0]) * 100
    semantic_score = max(0, min(100, semantic_score))
    
    final_score = (0.7 * keyword_score) + (0.3 * semantic_score)
    
    return round(final_score), matched_skills, missing_skills, sections