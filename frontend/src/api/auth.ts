import { apiClient } from "./client";
import type { LoginResponse, User } from "../types";

export async function registerUser(payload: { email: string; full_name: string; password: string }) {
  const response = await apiClient.post<User>("/auth/register", payload);
  return response.data;
}

export async function loginUser(payload: { email: string; password: string }) {
  const form = new URLSearchParams();
  form.set("username", payload.email);
  form.set("password", payload.password);
  const response = await apiClient.post<LoginResponse>("/auth/login", form, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  return response.data;
}

export async function fetchCurrentUser() {
  const response = await apiClient.get<User>("/auth/me");
  return response.data;
}
