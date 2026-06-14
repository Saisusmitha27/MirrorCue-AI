import { ChevronDown } from "lucide-react";
import { useState } from "react";
import clsx from "clsx";
import type { BiasFlag } from "../../types";

const badgeStyles: Record<BiasFlag["severity"], string> = {
  high: "bg-rose-100 text-rose-700 border-rose-200",
  medium: "bg-amber-100 text-amber-700 border-amber-200",
  low: "bg-emerald-100 text-emerald-700 border-emerald-200",
};

export function BiasCard({ flag }: { flag: BiasFlag }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <article className="animate-fade-in rounded-3xl border border-slate-200 bg-white shadow-sm">
      <button
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left hover:bg-slate-50 transition"
      >
        <div className="flex items-center gap-3 flex-1">
          <span className={clsx("rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide", badgeStyles[flag.severity])}>
            {flag.label} — {flag.severity}
          </span>
          <span className="text-sm text-slate-600">{flag.line_context}</span>
        </div>
        <ChevronDown className={clsx("h-5 w-5 text-slate-400 transition flex-shrink-0", expanded && "rotate-180")} />
      </button>

      {expanded ? (
        <div className="border-t border-slate-200 p-5 space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-600">📝 You Wrote</p>
              <p className="text-sm text-slate-900 font-medium">{flag.candidate_wrote}</p>
            </div>
            <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-rose-600">👁️ Recruiter Decoded in 7s</p>
              <p className="text-sm text-slate-900 italic">"{flag.recruiter_decoded}"</p>
            </div>
          </div>
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-600">💡 Bias Breaking Resolution</p>
            <p className="text-sm text-slate-900">{flag.fix}</p>
          </div>
        </div>
      ) : null}
    </article>
  );
}
