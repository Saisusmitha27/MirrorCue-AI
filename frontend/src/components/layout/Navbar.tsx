import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuthStore } from "../../store/authStore";

export function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { token, user, logout } = useAuthStore();

  const isAuthPage = ["/login", "/register"].includes(location.pathname);

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        {/* ── Logo + Tagline ── */}
        <Link to="/" className="flex items-center gap-3 group">
          <img src="/icon.png" alt="MirrorCue AI" className="h-11 w-11 rounded-xl object-cover shadow-md shadow-blue-200 transition group-hover:shadow-lg group-hover:shadow-blue-300 group-hover:scale-[1.03]" />
          <div>
            <p className="text-base font-bold text-slate-900 tracking-tight">MirrorCue AI</p>
            <p className="text-[11px] font-medium text-indigo-500 tracking-wide">
              Analyze <span className="text-slate-300">•</span> Understand <span className="text-slate-300">•</span> Improve
            </p>
          </div>
        </Link>

        {/* ── Right side ── */}
        <div className="flex items-center gap-3">
          {token && user ? (
            <>
              <span className="hidden items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50/80 px-4 py-1.5 text-sm font-semibold text-indigo-700 shadow-sm md:inline-flex">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-indigo-200 text-[11px] font-bold text-indigo-700">
                  {user.email?.split('@')[0]?.charAt(0).toUpperCase() || 'U'}
                </span>
                {user.full_name}
              </span>
              <button
                onClick={() => {
                  logout();
                  navigate("/login");
                }}
                className="flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-red-300 hover:bg-red-50 hover:text-red-600 hover:shadow-md"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
                  <polyline points="16 17 21 12 16 7" />
                  <line x1="21" y1="12" x2="9" y2="12" />
                </svg>
                Logout
              </button>
            </>
          ) : !isAuthPage ? (
            <>
              <Link to="/login" className="text-sm font-medium text-slate-600 transition hover:text-slate-900">
                Login
              </Link>
              <Link to="/register" className="rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-blue-200 transition hover:shadow-lg hover:shadow-blue-300 hover:from-blue-700 hover:to-indigo-700">
                Get Started
              </Link>
            </>
          ) : null}
        </div>
      </div>
    </header>
  );
}
