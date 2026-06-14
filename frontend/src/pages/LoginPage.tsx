import { Link } from "react-router-dom";
import { LoginForm } from "../components/auth/LoginForm";

export function LoginPage() {
  return (
    <div className="flex min-h-[calc(100vh-88px)] items-center justify-center px-6 py-10">
      <div className="w-full max-w-md rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl">
        <h1 className="text-3xl font-bold text-slate-900">Welcome back</h1>
        <p className="mt-2 text-slate-600">Sign in to continue your MirrorCue workflow.</p>
        <div className="mt-8">
          <LoginForm />
        </div>
        <p className="mt-6 text-center text-sm text-slate-600">
          Need an account?{" "}
          <Link className="text-teal font-medium hover:text-teal/90" to="/register">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
