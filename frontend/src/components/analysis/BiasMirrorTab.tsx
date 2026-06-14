import { useState } from "react";
import { ChevronDown, GraduationCap, Languages } from "lucide-react";
import clsx from "clsx";
import { BiasCard } from "./BiasCard";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";
import type { AnalysisResult } from "../../types";

export function BiasMirrorTab({ analysis }: { analysis: AnalysisResult }) {
  const bias = analysis.bias_result;
  const [branchExpanded, setBranchExpanded] = useState(true);
  const [masculineExpanded, setMasculineExpanded] = useState(true);

  if (!bias) {
    if (analysis.status === "failed") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold text-lg">Analysis Failed</p>
          <p className="mt-2 text-sm text-rose-600">We ran into an error while running the bias mirror pipeline.</p>
        </div>
      );
    }
    return (
      <ResumeAnalysisLoader
        title="Running Bias Audit..."
        phrases={[
          "Loading local XGBoost Multi-Label Classifier...",
          "Scanning demographic identifiers (name and gender cues)...",
          "Auditing college prestige and geographic location biases...",
          "Evaluating degree/branch priority vectors...",
          "Reviewing resume description credibility scores...",
        ]}
      />
    );
  }

  const getGradientColor = (score: number) => {
    if (score > 70) return "from-orange-400 to-rose-500";
    if (score > 40) return "from-amber-400 to-orange-500";
    return "from-emerald-400 to-teal-500";
  };

  const criticalCount = bias.flags.filter(f => f.severity === "high").length;
  const indiaSpecificCount = bias.flags.filter(
    f => f.bias_type && (f.bias_type.includes("india") || f.bias_type === "degree_branch_bias" || f.bias_type === "vernacular_english")
  ).length;

  const branch_bias = bias.branch_bias || null;
  const masculine_bias = bias.masculine_bias || null;

  const branchBiasRisk = branch_bias?.risk_level || "Low";
  const skillScore = branch_bias?.skill_alignment_score || 0;

  const masculineBiasRisk = masculine_bias?.risk_level || "Low";

  return (
    <div className="space-y-6">
      {/* Unconscious Bias Audit Summary */}
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-slate-900 mb-2">UNCONSCIOUS BIAS AUDIT REPORT</h3>
            <p className="text-sm text-slate-600">{bias.summary}</p>
          </div>
          <div className="text-right">
            <p className="text-sm uppercase tracking-wide text-slate-600 font-semibold">BIAS INDEX RATING</p>
            <p className="mt-2 text-4xl font-bold text-rose-600">
              {Math.round(bias.bias_score)}
              <span className="text-2xl text-slate-400">/100</span>
            </p>
          </div>
        </div>
        <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full bg-gradient-to-r ${getGradientColor(bias.bias_score)}`}
            style={{ width: `${Math.min(100, bias.bias_score)}%` }}
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-6 text-sm">
          <div>
            <p className="uppercase tracking-wide text-slate-600 font-semibold">LOCALIZED BIAS TRIGGERS</p>
            <p className="mt-1 text-lg font-bold text-slate-900">{indiaSpecificCount} India-Specific / Localized</p>
          </div>
          <div>
            <p className="uppercase tracking-wide text-slate-600 font-semibold">CRITICAL FAILURES</p>
            <p className="mt-1 text-lg font-bold text-rose-600">{criticalCount} High Severity</p>
          </div>
        </div>
      </section>

      {/* Enhanced Bias Categories Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Branch Bias Card */}
        <article className="animate-fade-in rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-indigo-50 p-3 text-indigo-600">
                  <GraduationCap className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Branch Bias Audit</h4>
                  <p className="text-xs text-slate-500">Evaluating degree-independent skill alignment</p>
                </div>
              </div>
              <button
                onClick={() => setBranchExpanded(prev => !prev)}
                className="rounded-xl p-1.5 hover:bg-slate-100 transition text-slate-400"
              >
                <ChevronDown className={clsx("h-5 w-5 transform transition", branchExpanded && "rotate-180")} />
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              <span
                className={clsx(
                  "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide border",
                  branchBiasRisk === "High" && "bg-rose-50 border-rose-200 text-rose-700",
                  branchBiasRisk === "Medium" && "bg-amber-50 border-amber-200 text-amber-700",
                  branchBiasRisk === "Low" && "bg-emerald-50 border-emerald-200 text-emerald-700"
                )}
              >
                Risk: {branchBiasRisk}
              </span>
              {branch_bias?.rankings_influenced && (
                <span className="rounded-full bg-rose-100 border border-rose-200 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-rose-700 animate-pulse">
                  ⚠️ Rankings Influenced
                </span>
              )}
            </div>

            <div className="flex items-center gap-4 rounded-2xl bg-slate-50 border border-slate-100 p-4">
              <div className="relative flex-shrink-0">
                <svg className="w-16 h-16 transform -rotate-90">
                  <circle cx="32" cy="32" r="28" className="text-slate-200" strokeWidth="5" stroke="currentColor" fill="transparent" />
                  <circle
                    cx="32"
                    cy="32"
                    r="28"
                    className="text-indigo-600 transition-all duration-500"
                    strokeWidth="5"
                    strokeDasharray={176}
                    strokeDashoffset={176 - (176 * skillScore) / 100}
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="transparent"
                  />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-slate-800">
                  {Math.round(skillScore)}%
                </span>
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">Skill Alignment Score</p>
                <p className="text-xs text-slate-500 leading-normal">
                  Purely candidate-centered technical competency, projects, certifications, assessments & GitHub activity (independent of college/branch).
                </p>
              </div>
            </div>

            {branchExpanded && (
              <div className="space-y-4 pt-2 border-t border-slate-100 text-sm">
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">📊 Evidence & Analysis</p>
                  <p className="text-slate-700 leading-relaxed italic bg-slate-50 p-3 rounded-2xl border border-slate-100">
                    {branch_bias?.evidence || "No evidence recorded."}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">💡 Confidence Level</p>
                  <p className="text-slate-700 font-medium">{branch_bias?.confidence || "Medium"}</p>
                </div>
                {branch_bias?.recommendations && branch_bias.recommendations.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">🌱 Bias Breaking Actions</p>
                    <ul className="list-disc pl-4 space-y-1.5 text-slate-700">
                      {branch_bias.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>

        {/* Masculine Language Bias Card */}
        <article className="animate-fade-in rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-pink-50 p-3 text-pink-600">
                  <Languages className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Masculine Language Audit</h4>
                  <p className="text-xs text-slate-500">Evaluating job description linguistic inclusivity</p>
                </div>
              </div>
              <button
                onClick={() => setMasculineExpanded(prev => !prev)}
                className="rounded-xl p-1.5 hover:bg-slate-100 transition text-slate-400"
              >
                <ChevronDown className={clsx("h-5 w-5 transform transition", masculineExpanded && "rotate-180")} />
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              <span
                className={clsx(
                  "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide border",
                  masculineBiasRisk === "High" && "bg-rose-50 border-rose-200 text-rose-700",
                  masculineBiasRisk === "Medium" && "bg-amber-50 border-amber-200 text-amber-700",
                  masculineBiasRisk === "Low" && "bg-emerald-50 border-emerald-200 text-emerald-700"
                )}
              >
                Risk: {masculineBiasRisk}
              </span>
              <span className="rounded-full bg-slate-100 border border-slate-200 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-700">
                {masculine_bias?.density_score || 0}% Density
              </span>
            </div>

            <div className="rounded-2xl bg-slate-50 border border-slate-100 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-600 mb-2">📊 Language Balance Index</p>
              <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    "h-full rounded-full transition-all duration-500",
                    masculineBiasRisk === "High" ? "bg-rose-500" : masculineBiasRisk === "Medium" ? "bg-amber-500" : "bg-emerald-500"
                  )}
                  style={{ width: `${Math.min(100, (masculine_bias?.density_score || 0) * 50)}%` }}
                />
              </div>
              <p className="text-[11px] text-slate-500 mt-1.5 leading-normal">
                Based on the ratio of gender-coded terms detected in the job description to the total word count.
              </p>
            </div>

            {masculineExpanded && (
              <div className="space-y-4 pt-2 border-t border-slate-100 text-sm">
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">📝 Evidence & Scan Results</p>
                  <p className="text-slate-700 leading-relaxed italic bg-slate-50 p-3 rounded-2xl border border-slate-100">
                    {masculine_bias?.evidence || "No evidence recorded."}
                  </p>
                </div>

                {masculine_bias?.matched_terms && masculine_bias.matched_terms.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">🔍 Configurable Dictionary Matches</p>
                    <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white">
                      <table className="w-full text-xs text-left border-collapse">
                        <thead className="bg-slate-50 text-slate-700 border-b border-slate-200 font-semibold">
                          <tr>
                            <th className="p-2.5 font-medium">Term Found</th>
                            <th className="p-2.5 font-medium">Inclusive Alternative</th>
                            <th className="p-2.5 font-medium text-center">Count</th>
                          </tr>
                        </thead>
                        <tbody>
                          {masculine_bias.matched_terms.map(item => (
                            <tr key={item.term} className="hover:bg-slate-50 border-b border-slate-100 last:border-0">
                              <td className="p-2.5 text-rose-600 font-mono font-semibold">{item.term}</td>
                              <td className="p-2.5 text-emerald-600 font-semibold">✓ {item.replacement}</td>
                              <td className="p-2.5 text-slate-500 text-center font-medium">{item.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {masculine_bias?.recommendation && (
                  <div className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">🌱 Inclusive Alternatives & Fixes</p>
                    <ul className="list-disc pl-4 space-y-1.5 text-slate-700">
                      {masculine_bias.recommendation.split("; ").map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </article>
      </div>

      {/* Individual Resume Bias Flag Cards */}
      <div className="space-y-4">
        {bias.flags.filter(flag => flag.bias_type !== "degree_branch_bias" && flag.bias_type !== "masculine_language_bias").length ? (
          bias.flags
            .filter(flag => flag.bias_type !== "degree_branch_bias" && flag.bias_type !== "masculine_language_bias")
            .map(flag => <BiasCard key={`${flag.bias_type}-${flag.candidate_wrote}`} flag={flag} />)
        ) : (
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-6 text-emerald-700">
            Great news — the current resume shows low visible bias signals.
          </div>
        )}
      </div>
    </div>
  );
}
