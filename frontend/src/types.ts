export interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  created_at: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ResumeUploadResponse {
  resume_id: string;
  analysis_id: string;
}

export interface AnalysisListItem {
  id: string;
  filename: string;
  ats_score?: number | null;
  bias_score?: number | null;
  created_at: string;
  status: string;
}

export interface BiasFlag {
  bias_type: string;
  label: string;
  candidate_wrote: string;
  recruiter_decoded: string;
  severity: "low" | "medium" | "high";
  fix: string;
  line_context: string;
  confidence?: "Low" | "Medium" | "High" | string;
  evidence?: string;
  skill_alignment_score?: number;
  rankings_influenced?: boolean;
  masculine_bias_density?: number;
  matched_terms?: Array<{ term: string; replacement: string; count: number }>;
}

export interface QAQuestion {
  id: string;
  section: string;
  item_name: string;
  question: string;
  why_needed: string;
  example_answer: string;
  answer_type: string;
}

export interface AnalysisResult {
  id: string;
  user_id: string;
  resume_id: string;
  jd_text: string;
  status: string;
  parsed_json?: Record<string, unknown> | null;
  ats_result?: {
    score: number;
    semantic_score: number;
    keyword_score: number;
    matched_keywords: string[];
    missing_keywords: string[];
    matched_keywords_detail?: Array<{ keyword: string; match_reason: string }>;
    missing_keywords_detail?: Array<{ keyword: string; importance: string }>;
    related_recommended_keywords?: Array<{ keyword: string; reason: string }>;
    additional_resume_strengths?: Array<{ item: string; category: string }>;
    formatting_flags: string[];
    jd_seniority_level: string;
    pass_threshold?: number;
    recommendation: string;
    breadth_bonus?: number | { points_added: number; matched_terms_count: number };
    section_breakdown?: {
      skills: { coverage_percent: number; semantic_similarity: number; weight: number };
      experience: { coverage_percent: number; semantic_similarity: number; weight: number };
      projects: { coverage_percent: number; semantic_similarity: number; weight: number };
      education: { coverage_percent: number; semantic_similarity: number; weight: number };
    };
  } | null;
  bias_result?: {
    flags: BiasFlag[];
    bias_score: number;
    summary: string;
    clean_signals: string[];
    india_specific_count: number;
    high_severity_count: number;
    branch_bias?: {
      risk_level: "Low" | "Medium" | "High";
      skill_alignment_score: number;
      severity: "low" | "medium" | "high";
      confidence: "Low" | "Medium" | "High";
      evidence: string;
      recommendations: string[];
      rankings_influenced: boolean;
    } | null;
    masculine_bias?: {
      risk_level: "Low" | "Medium" | "High";
      density_score: number;
      matched_terms: Array<{ term: string; replacement: string; count: number }>;
      severity: "low" | "medium" | "high";
      confidence: "Low" | "Medium" | "High";
      evidence: string;
      recommendation: string;
    } | null;
  } | null;
  qa_questions?: {
    questions: QAQuestion[];
  } | null;
  qa_answers?: Record<string, string> | null;
  rewrite_result?: {
    original_experience: Array<{ title: string; company: string; duration: string; bullets: string[] }>;
    original_projects: Array<{ name: string; tech_stack: string[]; bullets: string[] }>;
    rewritten_experience: Array<{ title: string; company: string; duration: string; bullets: string[]; keywords_added?: string[]; bias_phrases_removed?: string[] }>;
    rewritten_projects: Array<{ name: string; tech_stack: string[]; bullets: string[]; keywords_added?: string[] }>;
    rewritten_summary: string;
    ats_score_after: number;
    ats_score_delta?: number;
    total_keywords_added: number;
    total_bias_phrases_removed: number;
    changes_summary: string;
  } | null;
  created_at: string;
  updated_at: string;
}
