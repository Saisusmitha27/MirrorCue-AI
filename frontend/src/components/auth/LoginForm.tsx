import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loginUser } from "../../api/auth";
import { useAuthStore } from "../../store/authStore";

export function LoginForm() {
  const navigate = useNavigate();
  const setSession = useAuthStore((state) => state.setSession);
  const [form, setForm] = useState({ email: "", password: "" });

  const mutation = useMutation({
    mutationFn: () => loginUser(form),
    onSuccess: (data) => {
      setSession(data.access_token, data.user);
      navigate("/dashboard");
    },
  });

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        mutation.mutate();
      }}
      className="space-y-4"
    >
      <input
        className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-teal focus:ring-1 focus:ring-teal/20"
        placeholder="Email"
        type="email"
        value={form.email}
        onChange={(event) => setForm((current) => ({ ...current, email: event.target.value }))}
      />
      <input
        className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-teal focus:ring-1 focus:ring-teal/20"
        placeholder="Password"
        type="password"
        value={form.password}
        onChange={(event) => setForm((current) => ({ ...current, password: event.target.value }))}
      />
      {mutation.isError ? <p className="text-sm text-red-600">Login failed. Please check your credentials.</p> : null}
      <button className="w-full rounded-2xl bg-teal px-4 py-3 font-semibold text-white hover:bg-teal/90 transition" disabled={mutation.isPending}>
        {mutation.isPending ? "Signing in..." : "Login"}
      </button>
    </form>
  );
}
