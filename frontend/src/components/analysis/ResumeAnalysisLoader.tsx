import { useEffect, useState } from "react";

const DEFAULT_PHRASES = [
  "Extracting resume structure and text content...",
  "Running tokenization on skills & qualifications...",
  "Analyzing semantic match against job description...",
  "Scanning for potential name & gender bias indicators...",
  "Evaluating project credibility metrics...",
  "Checking education and tier-based parameters...",
  "Comparing keyword densities and alignments...",
  "Assembling final audit reports..."
];

interface ResumeAnalysisLoaderProps {
  title?: string;
  phrases?: string[];
}

export function ResumeAnalysisLoader({
  title = "Taking a closer look at your resume...",
  phrases = DEFAULT_PHRASES,
}: ResumeAnalysisLoaderProps) {
  const [phraseIdx, setPhraseIdx] = useState(0);
  const [progress, setProgress] = useState(10);

  useEffect(() => {
    if (phrases.length === 0) return;
    const phraseInterval = setInterval(() => {
      setPhraseIdx((prev) => (prev + 1) % phrases.length);
    }, 3500);

    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) return 95;
        return prev + Math.floor(Math.random() * 8) + 2;
      });
    }, 2500);

    return () => {
      clearInterval(phraseInterval);
      clearInterval(progressInterval);
    };
  }, [phrases]);

  return (
    <div className="flex flex-col items-center justify-center p-12 bg-white border border-slate-100 rounded-3xl shadow-xl max-w-xl mx-auto my-8 space-y-8 animate-fade-in">
      {/* Animated Document Icon */}
      <div className="relative flex items-center justify-center h-28 w-28 bg-emerald-50 rounded-2xl shadow-inner border border-emerald-100/50">
        <svg
          className="h-16 w-16 text-emerald-600 animate-pulse"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
          />
        </svg>
        <span className="absolute bottom-3 left-1/2 transform -translate-x-1/2 w-12 h-1 bg-emerald-500 rounded-full animate-bounce" />
      </div>

      <div className="text-center space-y-3">
        <h2 className="text-3xl font-extrabold text-slate-800 tracking-tight leading-snug">
          {title}
        </h2>
        {phrases.length > 0 ? (
          <p className="text-sm font-medium text-emerald-600 animate-pulse min-h-[20px]">
            {phrases[phraseIdx]}
          </p>
        ) : null}
      </div>

      {/* Progress Bar resembling the reference image */}
      <div className="w-full max-w-md bg-slate-100 h-2.5 rounded-full overflow-hidden border border-slate-200/50">
        <div
          className="bg-emerald-500 h-full rounded-full transition-all duration-1000 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      <p className="text-xs text-slate-400 font-medium">
        This usually takes 1–2 minutes.
      </p>
    </div>
  );
}
