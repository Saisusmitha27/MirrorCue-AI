import { Eye, EyeOff } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { loginUser, registerUser } from "../../api/auth";
import { useAuthStore } from "../../store/authStore";

export function RegisterForm() {
  const navigate = useNavigate();
  const setSession = useAuthStore((state) => state.setSession);
  const [form, setForm] = useState({
    full_name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [localError, setLocalError] = useState("");

  const mutation = useMutation({
    mutationFn: async () => {
      if (form.password !== form.confirmPassword) {
        throw new Error("Passwords do not match");
      }

      await registerUser({
        full_name: form.full_name,
        email: form.email,
        password: form.password,
      });

      return loginUser({
        email: form.email,
        password: form.password,
      });
    },
    onSuccess: (data) => {
      setLocalError("");
      setSession(data.access_token, data.user);
      navigate("/dashboard");
    },
    onError: (error: unknown) => {
      const message =
        error instanceof Error ? error.message : "Registration failed. Please try again.";
      setLocalError(message);
    },
  });

  const passwordsMatch = form.password === form.confirmPassword;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        setLocalError("");
        mutation.mutate();
      }}
      className="space-y-4"
    >
      <input
        className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-teal focus:ring-1 focus:ring-teal/20"
        placeholder="Full name"
        value={form.full_name}
        onChange={(event) => setForm((current) => ({ ...current, full_name: event.target.value }))}
      />
      <input
        className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-teal focus:ring-1 focus:ring-teal/20"
        placeholder="Email"
        type="email"
        value={form.email}
        onChange={(event) => setForm((current) => ({ ...current, email: event.target.value }))}
      />
      <div className="relative">
        <input
          className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 pr-12 text-slate-900 outline-none focus:border-teal focus:ring-1 focus:ring-teal/20"
          placeholder="Password"
          type={showPassword ? "text" : "password"}
          value={form.password}
          onChange={(event) => setForm((current) => ({ ...current, password: event.target.value }))}
        />
        <button
          type="button"
          onClick={() => setShowPassword((value) => !value)}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-600 transition hover:text-slate-900"
          aria-label={showPassword ? "Hide password" : "Show password"}
        >
          {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>
      <div className="relative">
        <input
          className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 pr-12 text-slate-900 outline-none focus:border-teal focus:ring-1 focus:ring-teal/20"
          placeholder="Re-enter password"
          type={showConfirmPassword ? "text" : "password"}
          value={form.confirmPassword}
          onChange={(event) => setForm((current) => ({ ...current, confirmPassword: event.target.value }))}
        />
        <button
          type="button"
          onClick={() => setShowConfirmPassword((value) => !value)}
          className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-600 transition hover:text-slate-900"
          aria-label={showConfirmPassword ? "Hide password" : "Show password"}
        >
          {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>
      {!passwordsMatch && form.confirmPassword ? (
        <p className="text-sm text-amber-600">Passwords do not match yet.</p>
      ) : null}
      {localError ? <p className="text-sm text-red-600">{localError}</p> : null}
      <button className="w-full rounded-2xl bg-teal px-4 py-3 font-semibold text-white hover:bg-teal/90 transition" disabled={mutation.isPending}>
        {mutation.isPending ? "Creating account..." : "Register"}
      </button>
      <p className="text-center text-sm text-slate-600">
        Already have an account?{" "}
        <Link className="text-teal font-medium hover:text-teal/90" to="/login">
          Login
        </Link>
      </p>
    </form>
  );
}
