import { apiClient } from "./client";
import type { ResumeUploadResponse } from "../types";

export async function uploadResume(file: File, jdText: string) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("jd_text", jdText);
  const response = await apiClient.post<ResumeUploadResponse>("/resume/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}
