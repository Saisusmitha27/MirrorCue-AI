import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchAnalysisList } from "../../api/analysis";
import { ResumeUploadCard } from "./ResumeUploadCard";

export function Dashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ["analysis-list"],
    queryFn: fetchAnalysisList,
  });

  return (
    <div className="grid gap-6 lg:grid-cols-[360px,1fr]">
      <ResumeUploadCard />

      <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Past Analyses</h2>
            <p className="text-sm text-slate-600">Review ATS scores, bias risk, and rewrites.</p>
          </div>
        </div>

        {isLoading ? (
          <div className="grid gap-4">
            {Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="h-28 animate-pulse rounded-2xl bg-slate-200" />
            ))}
          </div>
        ) : (
          <div className="grid gap-4">
            {data?.length ? (
              data.map((item) => (
                <div key={item.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4 hover:shadow-md transition">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="font-medium text-slate-900">{item.filename}</h3>
                      <p className="text-sm text-slate-600">{new Date(item.created_at).toLocaleString()}</p>
                    </div>
                    <span className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-700 bg-white">
                      {item.status}
                    </span>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs text-emerald-700">
                      ATS: {item.ats_score?.toFixed(0) ?? "--"}
                    </span>
                    <span className="rounded-full bg-rose-100 px-3 py-1 text-xs text-rose-700">
                      Bias: {item.bias_score?.toFixed(0) ?? "--"}
                    </span>
                  </div>
                  <Link
                    to={`/analysis/${item.id}`}
                    className="mt-4 inline-flex rounded-xl border border-blue-300 bg-blue-50 px-4 py-2 text-sm text-blue-600 hover:bg-blue-100 transition"
                  >
                    View Results
                  </Link>
                </div>
              ))
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-300 p-8 text-center text-slate-500 bg-slate-50">
                No analyses yet. Upload your first resume to get started.
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
