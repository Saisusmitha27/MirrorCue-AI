import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { fetchAnalysis } from "../../api/analysis";
import { ATSTab } from "./ATSTab";
import { BiasMirrorTab } from "./BiasMirrorTab";
import { RewriteTab } from "./RewriteTab";
import { ResumeAnalysisLoader } from "./ResumeAnalysisLoader";

const tabs = ["ATS Report", "Bias Mirror", "Smart Rewrite"] as const;

export function AnalysisPage() {
  const { id = "" } = useParams();
  const [activeTab, setActiveTab] = useState<(typeof tabs)[number]>("ATS Report");

  const { data, isLoading } = useQuery({
    queryKey: ["analysis", id],
    queryFn: () => fetchAnalysis(id),
    enabled: Boolean(id),
    refetchInterval: (query) => {
      const analysis = query.state.data;
      if (!analysis) return 3000;
      const terminalStatuses = ["complete", "failed"];
      const activeStatuses = ["running", "rewriting", "rewrite", "qa_validate", "qa_validated"];
      if (activeStatuses.includes(analysis.status) || !terminalStatuses.includes(analysis.status)) {
        return 3000;
      }
      return analysis.rewrite_result ? false : 1500;
    },
  });

  if (isLoading || !data) {
    return (
      <div className="grid gap-4">
        {Array.from({ length: 3 }).map((_, index) => (
          <div key={index} className="h-32 animate-pulse rounded-3xl bg-slate-300" />
        ))}
      </div>
    );
  }

  const isActive = !["complete", "failed"].includes(data.status);

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Analysis Progress</h1>
            <p className="text-sm text-slate-600">Current stage: {data.status}</p>
          </div>
          <div className="rounded-full border border-blue-300 bg-blue-50 px-4 py-2 text-sm text-blue-600 font-medium">
            {isActive ? "Analysis in Progress" : "Results Ready"}
          </div>
        </div>
      </section>

      <div className="flex flex-wrap gap-3">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`rounded-full px-4 py-2 text-sm font-medium transition ${
              activeTab === tab ? "bg-blue-600 text-white" : "border border-slate-300 text-slate-700 hover:bg-slate-50"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === "ATS Report" ? <ATSTab analysis={data} /> : null}
      {activeTab === "Bias Mirror" ? <BiasMirrorTab analysis={data} /> : null}
      {activeTab === "Smart Rewrite" ? <RewriteTab analysis={data} /> : null}
    </div>
  );
}
