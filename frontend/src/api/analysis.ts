import { apiClient } from "./client";
import type { AnalysisListItem, AnalysisResult } from "../types";

export async function runAnalysis(analysisId: string) {
  const response = await apiClient.post<{ analysis_id: string; status: string }>("/analysis/run", {
    analysis_id: analysisId,
  });
  return response.data;
}

export async function fetchAnalysis(analysisId: string) {
  const response = await apiClient.get<AnalysisResult>(`/analysis/${analysisId}`);
  return response.data;
}

export async function fetchAnalysisList() {
  const response = await apiClient.get<AnalysisListItem[]>("/analysis/list");
  return response.data;
}

export async function submitRewriteAnswers(analysisId: string, qaAnswers: Record<string, string>) {
  const response = await apiClient.post<{ status: string }>(`/analysis/${analysisId}/rewrite`, {
    qa_answers: qaAnswers,
  });
  return response.data;
}
