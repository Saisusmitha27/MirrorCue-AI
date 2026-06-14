import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { submitRewriteAnswers } from "../../api/analysis";
import type { QAQuestion } from "../../types";

export function QAForm({ analysisId, questions }: { analysisId: string; questions: QAQuestion[] }) {
  const queryClient = useQueryClient();
  const [answers, setAnswers] = useState<Record<string, string>>(
    Object.fromEntries(questions.map((question) => [question.id, ""])),
  );

  const canSubmit = useMemo(
    () => questions.every((question) => (answers[question.id] || "").trim().length > 0),
    [answers, questions],
  );

  const mutation = useMutation({
    mutationFn: () => submitRewriteAnswers(analysisId, answers),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["analysis", analysisId] });
    },
  });

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-blue-200 bg-blue-50 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">VERIFIED SDE RESPONSE GATE</h3>
            <p className="mt-2 text-sm text-slate-700">MirrorCue AI refuses to guess or fabricate metrics. Provide brief quantitative filters to unlock SDE copy block.</p>
          </div>
          <span className="inline-flex items-center gap-1 rounded-full bg-blue-600 text-white px-3 py-1 text-xs font-bold uppercase tracking-wide flex-shrink-0">
            ✓ NO HALLUCINATIONS
          </span>
        </div>
      </section>

      {questions.map((question, idx) => (
        <div key={question.id} className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-center gap-3">
            <span className="inline-flex items-center justify-center h-8 w-8 rounded-full bg-blue-100 text-blue-600 font-bold text-sm">Q{idx + 1}</span>
            <div className="flex-1">
              <p className="text-xs uppercase tracking-wide font-semibold text-slate-600">TARGET: {question.section}</p>
              <h4 className="text-base font-semibold text-slate-900 mt-1">{question.question}</h4>
            </div>
          </div>
          
          <div className="bg-slate-50 rounded-2xl p-4 mb-4 border border-slate-200">
            <p className="text-xs uppercase tracking-wide font-semibold text-slate-600 mb-2">CORE CONTEXT</p>
            <p className="text-sm text-slate-700">{question.why_needed}</p>
          </div>
          
          {question.example_answer && (
            <div className="bg-blue-50 rounded-2xl p-4 mb-4 border border-blue-200">
              <p className="text-xs uppercase tracking-wide font-semibold text-blue-600 mb-2">GUIDE</p>
              <p className="text-sm text-slate-700">{question.example_answer}</p>
            </div>
          )}
          
          <textarea
            className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200 min-h-32"
            placeholder="Your answer here..."
            value={answers[question.id] ?? ""}
            onChange={(event) => setAnswers((current) => ({ ...current, [question.id]: event.target.value }))}
          />
        </div>
      ))}

      <button
        onClick={() => mutation.mutate()}
        disabled={!canSubmit || mutation.isPending}
        className="w-full rounded-2xl bg-blue-600 px-4 py-3 font-semibold text-white hover:bg-blue-700 disabled:opacity-60 transition"
      >
        {mutation.isPending ? "Generating rewrite..." : "Submit Answers & Generate Rewrite"}
      </button>
    </div>
  );
}
