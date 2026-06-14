import type { AnalysisResult } from "../../types";
import { QAForm } from "./QAForm";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(href);
}

export function RewriteTab({ analysis }: { analysis: AnalysisResult }) {
  const questions = analysis.qa_questions?.questions ?? [];
  const rewrite = analysis.rewrite_result;

  // Extract bias flags once to avoid double-filtering and fix the TS conditional error
  const biasFlags = analysis.bias_result?.flags?.filter((f: any) =>
    f.candidate_wrote &&
    f.candidate_wrote !== "resume content" &&
    f.candidate_wrote !== "project details" &&
    f.candidate_wrote !== "experience details"
  ) ?? [];

  if (!rewrite) {
    if (analysis.status === "failed") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold text-lg">Analysis Failed</p>
          <p className="mt-2 text-sm text-rose-600">We ran into an error while running the resume rewrite pipeline.</p>
        </div>
      );
    }

    // QA form takes priority — if questions exist, always show the form first
    if (questions.length) {
      if (
        analysis.status === "rewriting" ||
        analysis.status === "rewrite" ||
        analysis.status === "qa_validate" ||
        analysis.status === "qa_validated"
      ) {
        return (
          <ResumeAnalysisLoader
            title="Generating Verified Resume Rewrite..."
            phrases={[
              "Mapping Q&A answers to resume items...",
              "Resolving grammatical structures...",
              "Injecting missing ATS keywords...",
              "Formulating action-oriented bullets with metrics...",
              "Removing unconscious bias phrases...",
            ]}
          />
        );
      }
      return (
        <QAForm analysisId={analysis.id} questions={questions} />
      );
    }

    // Only show error if status is complete AND there are no questions to answer
    if (analysis.status === "complete") {
      return (
        <div className="rounded-2xl border border-dashed border-rose-300 bg-rose-50 p-8 text-center text-rose-600">
          <p className="font-semibold">Rewrite Not Available</p>
          <p className="mt-2 text-sm text-rose-600">The analysis completed but the rewrite data wasn't generated. Please try submitting the QA answers again.</p>
        </div>
      );
    }

    // Default: pipeline still running, questions not yet generated
    return (
      <ResumeAnalysisLoader
        title="Formulating Clarification Questions..."
        phrases={[
          "Scanning resume sections for metric and technology gaps...",
          "Comparing keyword gaps for specialized questions...",
          "Preparing targeted clarification questions...",
        ]}
      />
    );
  }

  const exportText = [
    "MirrorCue Rewrite Summary",
    rewrite.rewritten_summary || "Summary not available",
    "",
    "Experience",
    ...rewrite.rewritten_experience?.flatMap((item) => [
      `${item.title} | ${item.company} | ${item.duration}`,
      ...item.bullets.map((bullet) => `- ${bullet}`),
      "",
    ]) || ["No experience items"],
    "Projects",
    ...rewrite.rewritten_projects?.flatMap((item) => [
      `${item.name} | ${item.tech_stack.join(", ")}`,
      ...item.bullets.map((bullet) => `- ${bullet}`),
      "",
    ]) || ["No projects"],
  ].join("\n");

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap gap-8 items-end">
            <div>
              <p className="text-sm text-slate-500 uppercase tracking-wide font-semibold mb-1">ATS Score</p>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-slate-400">{Math.round(analysis.ats_result?.score ?? 0)}%</span>
                <span className="text-slate-400 text-xl">→</span>
                <span className="text-4xl font-bold text-emerald-600">{Math.round(rewrite.ats_score_after)}%</span>
                {rewrite.ats_score_delta != null && (
                  <span className={`text-lg font-semibold ${rewrite.ats_score_delta >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                    ({rewrite.ats_score_delta >= 0 ? "+" : ""}{Math.round(rewrite.ats_score_delta)}%)
                  </span>
                )}
              </div>
            </div>
            <div>
              <p className="text-sm text-slate-500 uppercase tracking-wide font-semibold mb-1">Keywords added</p>
              <p className="text-3xl font-bold text-blue-600">{rewrite.total_keywords_added ?? 0}</p>
            </div>
            <div>
              <p className="text-sm text-slate-500 uppercase tracking-wide font-semibold mb-1">Bias phrases removed</p>
              <p className="text-3xl font-bold text-purple-600">{rewrite.total_bias_phrases_removed ?? 0}</p>
            </div>
          </div>
          <button
            onClick={() => downloadText("mirrorcue-rewrite.txt", exportText)}
            className="rounded-2xl border border-blue-300 bg-blue-50 px-4 py-2 text-sm text-blue-600 hover:bg-blue-100 transition self-start"
          >
            Download Rewrite
          </button>
        </div>
      </section>

      {/* Bias Removal Evidence Panel */}
      {biasFlags.length > 0 && (
        <section className="rounded-3xl border border-purple-200 bg-purple-50 p-5">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-purple-700 mb-3">
            Bias phrases addressed in this rewrite
          </h3>
          <div className="space-y-2">
            {biasFlags.map((flag: any, i: number) => (
              <div key={i} className="flex flex-wrap items-start gap-2 text-sm">
                <span className="rounded-lg bg-rose-100 text-rose-700 px-2 py-0.5 line-through decoration-rose-400">
                  {flag.candidate_wrote}
                </span>
                <span className="text-slate-400 mt-0.5">→</span>
                <span className="rounded-lg bg-emerald-100 text-emerald-700 px-2 py-0.5">
                  {flag.fix || "rephrased with action verbs"}
                </span>
                <span className="text-slate-400 text-xs mt-1">
                  ({flag.label || flag.bias_type})
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-3xl border border-rose-200 bg-rose-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">Original</h3>
          <div className="space-y-4">
            {rewrite.original_experience?.length ? (
              rewrite.original_experience.map((item, index) => (
                <div key={`${item.title}-${index}`} className="rounded-2xl bg-white border border-rose-200 p-4">
                  <p className="font-medium text-slate-900">{item.title}</p>
                  <p className="text-sm text-slate-600">
                    {item.company}
                    {item.duration && item.duration !== "Not specified" && item.duration !== "N/A"
                      ? ` · ${item.duration}`
                      : ""}
                  </p>
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets.map((bullet) => <li key={bullet}>• {bullet}</li>)}
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No experience items</p>
            )}
            {rewrite.original_projects?.length ? (
              rewrite.original_projects.map((item, index) => (
                <div key={`${item.name}-${index}`} className="rounded-2xl bg-white border border-rose-200 p-4">
                  <p className="font-medium text-slate-900">{item.name}</p>
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets.map((bullet) => <li key={bullet}>• {bullet}</li>)}
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No projects</p>
            )}
          </div>
        </section>

        <section className="rounded-3xl border border-emerald-200 bg-emerald-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-slate-900 uppercase tracking-wider">MirrorCue Rewrite</h3>
          <div className="space-y-4">
            {rewrite.rewritten_experience?.length ? (
              rewrite.rewritten_experience.map((item, index) => (
                <div key={`${item.title}-${index}`} className="rounded-2xl bg-white border border-emerald-200 p-4">
                  <p className="font-medium text-slate-900">{item.title}</p>
                  {(item.company || (item.duration && item.duration !== "Not specified")) && (
                    <p className="text-sm text-slate-600">
                      {item.company}
                      {item.duration && item.duration !== "Not specified" && item.duration !== "N/A"
                        ? ` · ${item.duration}`
                        : ""}
                    </p>
                  )}
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets?.length
                      ? item.bullets.filter(Boolean).map((bullet, bi) => (
                          <li key={bi} className="flex gap-2">
                            <span className="text-emerald-500 mt-0.5">•</span>
                            <span>{bullet}</span>
                          </li>
                        ))
                      : <li className="text-slate-400 italic">No bullets generated for this item.</li>
                    }
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No rewritten experience</p>
            )}
            {rewrite.rewritten_projects?.length ? (
              rewrite.rewritten_projects.map((item, index) => (
                <div key={`${item.name}-${index}`} className="rounded-2xl bg-white border border-emerald-200 p-4">
                  <p className="font-medium text-slate-900">{item.name}</p>
                  {item.tech_stack?.length > 0 && (
                    <p className="text-sm text-slate-600">{item.tech_stack.filter(Boolean).join(", ")}</p>
                  )}
                  <ul className="mt-3 space-y-2 text-sm text-slate-700">
                    {item.bullets?.length
                      ? item.bullets.filter(Boolean).map((bullet, bi) => (
                          <li key={bi} className="flex gap-2">
                            <span className="text-emerald-500 mt-0.5">•</span>
                            <span>{bullet}</span>
                          </li>
                        ))
                      : <li className="text-slate-400 italic">No bullets generated for this item.</li>
                    }
                  </ul>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No rewritten projects</p>
            )}
          </div>
        </section>
      </div>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900 uppercase tracking-wider">New Summary</h3>
        <p className="mt-3 text-slate-700">{rewrite.rewritten_summary || "Summary not available"}</p>
      </section>

      <details className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm group">
        <summary className="cursor-pointer text-lg font-semibold text-slate-900 group-open:text-blue-600 uppercase tracking-wider">Changes Made</summary>
        <p className="mt-4 text-slate-700">{rewrite.changes_summary || "Changes summary not available"}</p>
      </details>
    </div>
  );
}