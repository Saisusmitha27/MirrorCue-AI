import { create } from "zustand";
import type { User } from "../types";

interface AuthState {
  token: string | null;
  user: User | null;
  hydrate: () => void;
  setSession: (token: string, user: User) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  user: null,
  hydrate: () =>
    set({
      token: localStorage.getItem("mirrorcue_token"),
      user: JSON.parse(localStorage.getItem("mirrorcue_user") || "null"),
    }),
  setSession: (token, user) => {
    localStorage.setItem("mirrorcue_token", token);
    localStorage.setItem("mirrorcue_user", JSON.stringify(user));
    set({ token, user });
  },
  logout: () => {
    localStorage.removeItem("mirrorcue_token");
    localStorage.removeItem("mirrorcue_user");
    set({ token: null, user: null });
  },
}));
