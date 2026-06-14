import type { AnalysisResult } from "../../types";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";

function Gauge({ score }: { score: number }) {
  const color = score >= 70 ? "#22c55e" : score >= 49 ? "#f59e0b" : "#ef4444";
  const matchLabel = score >= 70 ? "Good match" : score >= 49 ? "Moderate match" : "Needs work";
  return (
    <div
      className="mx-auto flex h-44 w-44 items-center justify-center rounded-full"
      style={{
        background: `conic-gradient(${color} ${score * 3.6}deg, #f1f5f9 0deg)`,
        boxShadow: "0 0 30px rgba(0,212,170,0.1)",
      }}
    >
      <div className="flex h-32 w-32 flex-col items-center justify-center rounded-full bg-white text-slate-900">
        <span className="text-4xl font-bold">{Math.round(score)}%</span>
        <span className="text-xs uppercase tracking-widest text-slate-500">ATS Match</span>
        <span className="mt-1 text-[10px] font-semibold uppercase tracking-wide text-slate-400">{matchLabel}</span>
      </div>
    </div>
  );
}

export function ATSTab({ analysis }: { analysis: AnalysisResult }) {
  const result = analysis.ats_result;
  if (!result) {
    if (analysis.status === "failed") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold text-lg">Analysis Failed</p>
          <p className="mt-2 text-sm text-rose-600">We ran into an error while running the ATS matcher pipeline.</p>
        </div>
      );
    }
    return (
      <ResumeAnalysisLoader
        title="Analyzing ATS Alignment..."
        phrases={[
          "Parsing job description requirements...",
          "Comparing resume skill set mapping...",
          "Calculating semantic similarity metrics...",
          "Running term-frequency analysis...",
        ]}
      />
    );
  }

  const semanticScore = result.semantic_score || 0;
  const keywordScore = result.keyword_score || 0;

  const matchedDetail =
    result.matched_keywords_detail?.length
      ? result.matched_keywords_detail
      : result.matched_keywords.map((keyword) => ({
          keyword,
          match_reason: "Aligned with job description requirements",
        }));

  const missingDetail =
    result.missing_keywords_detail?.length
      ? result.missing_keywords_detail
      : result.missing_keywords.map((keyword) => ({
          keyword,
          importance: "Present in job description but missing from resume",
        }));

  const relatedKeywords = result.related_recommended_keywords ?? [];
  const resumeStrengths = result.additional_resume_strengths ?? [];

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-6 text-lg font-semibold uppercase tracking-wider text-slate-600">
          ATS Alignment Index
        </h3>
        <div className="flex flex-col items-center gap-6">
          <Gauge score={result.score} />
          <div className="w-full grid grid-cols-2 gap-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-center">
              <p className="text-sm uppercase tracking-wide text-slate-600 font-semibold">Semantic Match</p>
              <p className="mt-2 text-2xl font-bold text-slate-900">{Math.round(semanticScore)}%</p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-center">
              <p className="text-sm uppercase tracking-wide text-slate-600 font-semibold">Keyword Match</p>
              <p className="mt-2 text-2xl font-bold text-slate-900">{Math.round(keywordScore)}%</p>
            </div>
          </div>

          {result.section_breakdown ? (
            <div className="w-full mt-2 border-t border-slate-100 pt-6">
              <h4 className="mb-4 text-xs font-bold uppercase tracking-wider text-slate-500 text-center">
                Detailed Section Breakdown
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
                {(["skills", "experience", "projects", "education"] as const).map((sec) => {
                  const info = result.section_breakdown?.[sec];
                  if (!info) return null;
                  return (
                    <div key={sec} className="rounded-2xl border border-slate-200 bg-slate-50/50 p-4 transition-all hover:bg-slate-50 w-full text-left">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-semibold capitalize text-slate-800">{sec}</p>
                        <span className="rounded-lg bg-slate-200 px-2 py-0.5 text-xs font-bold text-slate-600">
                          {info.weight}x wt
                        </span>
                      </div>
                      <div className="mt-3 space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">Keyword Match:</span>
                          <span className="font-bold text-slate-700">{Math.round(info.coverage_percent)}%</span>
                        </div>
                        <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-amber-500 h-full rounded-full" style={{ width: `${info.coverage_percent}%` }} />
                        </div>

                        <div className="flex justify-between text-xs pt-1">
                          <span className="text-slate-500">Semantic Match:</span>
                          <span className="font-bold text-slate-700">{Math.round(info.semantic_similarity)}%</span>
                        </div>
                        <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-blue-500 h-full rounded-full" style={{ width: `${info.semantic_similarity}%` }} />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <section className="rounded-3xl border border-emerald-200 bg-emerald-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
            Matched Keywords ({matchedDetail.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {matchedDetail.length ? (
              matchedDetail.map((item) => (
                <span
                  key={item.keyword}
                  className="inline-flex items-center rounded-xl border border-emerald-200 bg-white px-3 py-1.5 text-sm font-medium text-emerald-800 shadow-sm"
                >
                  <span className="text-emerald-500 mr-1.5">✓</span>
                  {item.keyword}
                </span>
              ))
            ) : (
              <p className="text-sm text-slate-600">No matched keywords detected yet.</p>
            )}
          </div>
        </section>

        <section className="rounded-3xl border border-rose-200 bg-rose-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
            Missing Keywords ({missingDetail.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {missingDetail.length ? (
              missingDetail.map((item) => (
                <span
                  key={item.keyword}
                  className="inline-flex items-center rounded-xl border border-rose-200 bg-white px-3 py-1.5 text-sm font-medium text-rose-800 shadow-sm"
                >
                  <span className="text-rose-500 mr-1.5">✕</span>
                  {item.keyword}
                </span>
              ))
            ) : (
              <p className="text-sm text-slate-400">No critical gaps detected.</p>
            )}
          </div>
        </section>
      </div>

      <section className="rounded-3xl border border-blue-200 bg-blue-50 p-6">
        <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
          Related Recommended Keywords ({relatedKeywords.length})
        </h3>
        <p className="mb-4 text-sm text-slate-600">
          Industry-relevant terms recruiters and ATS systems commonly expect for this role.
        </p>
        <div className="space-y-3">
          {relatedKeywords.length ? (
            relatedKeywords.map((item) => (
              <div
                key={item.keyword}
                className="rounded-2xl border border-blue-200 bg-white px-4 py-3"
              >
                <p className="font-medium text-blue-800">{item.keyword}</p>
                {item.reason ? (
                  <p className="mt-1 text-sm text-slate-600">{item.reason}</p>
                ) : null}
              </div>
            ))
          ) : (
            <p className="text-sm text-slate-500">No market recommendations generated.</p>
          )}
        </div>
      </section>

      <section className="rounded-3xl border border-violet-200 bg-violet-50 p-6">
        <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
          Additional Resume Strengths ({resumeStrengths.length})
        </h3>
        <p className="mb-4 text-sm text-slate-600">
          Skills, certifications, and achievements in your resume not explicitly required by the JD.
        </p>
        <div className="flex flex-wrap gap-3">
          {resumeStrengths.length ? (
            resumeStrengths.map((item) => (
              <span
                key={`${item.item}-${item.category}`}
                className="inline-flex flex-col rounded-2xl border border-violet-200 bg-white px-4 py-2 text-sm"
              >
                <span className="font-medium text-violet-900">{item.item}</span>
                {item.category ? (
                  <span className="text-xs uppercase tracking-wide text-violet-600 mt-1">
                    {item.category}
                  </span>
                ) : null}
              </span>
            ))
          ) : (
            <p className="text-sm text-slate-500">No extra strengths identified.</p>
          )}
        </div>
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">
          Formatting Flags
        </h3>
        <div className="space-y-3">
          {result.formatting_flags.length ? (
            result.formatting_flags.map((flag) => (
              <div
                key={flag}
                className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"
              >
                {flag}
              </div>
            ))
          ) : (
            <p className="text-sm text-slate-500">No formatting warnings.</p>
          )}
        </div>
        <div className="mt-6 rounded-2xl bg-slate-50 border border-slate-200 p-4">
          <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Recommendation</p>
          <p className="mt-2 text-slate-900">{result.recommendation}</p>
        </div>
      </section>
    </div>
  );
}