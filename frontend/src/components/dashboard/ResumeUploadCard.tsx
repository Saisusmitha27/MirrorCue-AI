import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import { runAnalysis } from "../../api/analysis";
import { uploadResume } from "../../api/resume";

export function ResumeUploadCard() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jdText, setJdText] = useState("");

  const mutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile) {
        throw new Error("Please add a PDF resume");
      }
      const upload = await uploadResume(selectedFile, jdText);
      await runAnalysis(upload.analysis_id);
      return upload.analysis_id;
    },
    onSuccess: (analysisId) => navigate(`/analysis/${analysisId}`),
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    onDrop: (files) => setSelectedFile(files[0] ?? null),
  });

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-slate-900">New Analysis</h3>
        <p className="text-sm text-slate-600">Upload a resume, paste the JD, and let the pipeline run.</p>
      </div>

      <div
        {...getRootProps()}
        className={`cursor-pointer rounded-2xl border border-dashed p-6 text-center transition ${
          isDragActive ? "border-blue-400 bg-blue-50" : "border-slate-300 bg-slate-50"
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-sm text-slate-600">{selectedFile ? selectedFile.name : "Drag a PDF here or click to upload"}</p>
      </div>

      <textarea
        className="mt-4 min-h-40 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200"
        placeholder="Paste the job description here..."
        value={jdText}
        onChange={(event) => setJdText(event.target.value)}
      />

      {mutation.isError ? <p className="mt-3 text-sm text-red-600">We hit an upload issue. Please check the file and JD text.</p> : null}

      <button
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending || !selectedFile || !jdText.trim()}
        className="mt-4 w-full rounded-2xl bg-blue-600 px-4 py-3 font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition"
      >
        {mutation.isPending ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}
