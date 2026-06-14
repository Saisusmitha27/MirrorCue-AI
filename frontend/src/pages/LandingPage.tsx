import { useState } from "react";
import { Link } from "react-router-dom";

function IconDoc() {
  return (
    <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="#3b82f6" strokeWidth={1.6}>
      <rect x="4" y="2" width="16" height="20" rx="2" />
      <path d="M8 7h8M8 11h8M8 15h5" strokeLinecap="round" />
    </svg>
  );
}
function IconBolt() {
  return (
    <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="#3b82f6" strokeWidth={1.6}>
      <path d="M13 2L4.5 13.5H12L11 22l8.5-11.5H12.5L13 2z" strokeLinejoin="round" />
    </svg>
  );
}
function IconShield() {
  return (
    <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="#3b82f6" strokeWidth={1.6}>
      <path d="M12 2l8 4v5c0 5-3.5 9-8 11-4.5-2-8-6-8-11V6l8-4z" strokeLinejoin="round" />
      <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function IconStar() {
  return (
    <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="#7c3aed" strokeWidth={1.8}>
      <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6L12 2z" strokeLinejoin="round" />
    </svg>
  );
}
function IconFile() {
  return (
    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="#7c3aed" strokeWidth={1.8}>
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}
function IconCheck() {
  return (
    <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="#3b82f6" strokeWidth={2.5}>
      <path d="M20 6L9 17l-5-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ArcGauge({ score, color, trackColor }: { score: number; color: string; trackColor: string }) {
  const r = 52;
  const cx = 64;
  const cy = 64;
  const circumference = Math.PI * r;
  const dash = (score / 100) * circumference;

  return (
    <svg width="128" height="76" viewBox="0 0 128 76" style={{ flexShrink: 0 }}>
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke={trackColor} strokeWidth="11" strokeLinecap="round"
      />
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke={color} strokeWidth="11" strokeLinecap="round"
        strokeDasharray={`${dash} ${circumference}`}
      />
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize="24" fontWeight="700"
        fill={color} fontFamily="Inter, system-ui, sans-serif">{score}</text>
      <text x={cx} y={cx + 8} textAnchor="middle" fontSize="11" fill="#94a3b8"
        fontFamily="Inter, system-ui, sans-serif">/100</text>
    </svg>
  );
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div style={{ height: 4, background: "#e2e8f0", borderRadius: 99, overflow: "hidden", marginTop: 8 }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 99 }} />
    </div>
  );
}

type Tab = "ats" | "bias" | "rewrite";

function PreviewCard() {
  const [tab, setTab] = useState<Tab>("ats");

  const tabs: { id: Tab; icon: string; label: string }[] = [
    { id: "ats", icon: "📋", label: "ATS Score" },
    { id: "bias", icon: "🛡", label: "Bias Mirror" },
    { id: "rewrite", icon: "✦", label: "Rewrite Engine" },
  ];

  return (
    <div style={{
      background: "#fff",
      borderRadius: 20,
      border: "1px solid #e2e8f0",
      boxShadow: "0 4px 24px rgba(0,0,0,0.07)",
      overflow: "hidden",
      height: "100%",
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Tab bar */}
      <div style={{ display: "flex", borderBottom: "1px solid #f1f5f9", padding: "0 8px" }}>
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: 1,
            padding: "18px 8px",
            fontSize: 14,
            fontWeight: tab === t.id ? 600 : 500,
            color: tab === t.id ? "#3b82f6" : "#94a3b8",
            background: "none",
            border: "none",
            borderBottom: tab === t.id ? "2px solid #3b82f6" : "2px solid transparent",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
            marginBottom: -1,
            whiteSpace: "nowrap",
          }}>
            <span style={{ fontSize: 15 }}>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "0 28px", flex: 1, display: "flex", flexDirection: "column" }}>
        {tab === "ats" && (
          <>
            {/* ATS Score row */}
            <div style={{
              display: "flex", alignItems: "center", gap: 24,
              padding: "28px 0", borderBottom: "1px solid #f1f5f9"
            }}>
              <ArcGauge score={82} color="#3b82f6" trackColor="#dbeafe" />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", margin: "0 0 4px" }}>ATS Score</p>
                <p style={{ fontSize: 14, fontWeight: 600, color: "#3b82f6", margin: "0 0 8px" }}>Good chance of passing ATS</p>
                <p style={{ fontSize: 13, color: "#64748b", lineHeight: 1.6, margin: 0 }}>
                  Your resume is well-structured and matches key ATS criteria. Some improvements can increase the score.
                </p>
                <Bar pct={82} color="#3b82f6" />
              </div>
            </div>

            {/* Bias Score row */}
            <div style={{
              display: "flex", alignItems: "center", gap: 24,
              padding: "28px 0", borderBottom: "1px solid #f1f5f9"
            }}>
              <ArcGauge score={68} color="#f59e0b" trackColor="#fef3c7" />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", margin: "0 0 4px" }}>Bias Score</p>
                <p style={{ fontSize: 14, fontWeight: 600, color: "#f59e0b", margin: "0 0 8px" }}>Moderate bias risk</p>
                <p style={{ fontSize: 13, color: "#64748b", lineHeight: 1.6, margin: 0 }}>
                  We detected patterns that may unconsciously influence recruiter perception.
                </p>
                <Bar pct={68} color="#f59e0b" />
              </div>
            </div>

            {/* Purple CTA strip */}
            <div style={{
              margin: "0 -28px",
              background: "linear-gradient(135deg, #f5f3ff, #ede9fe)",
              padding: "22px 28px",
              display: "flex",
              alignItems: "center",
              gap: 16,
              marginTop: "auto",
            }}>
              <div style={{
                width: 48, height: 48, borderRadius: 12,
                background: "#ddd6fe",
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
              }}>
                <IconStar />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 15, fontWeight: 700, color: "#6d28d9", margin: "0 0 4px" }}>
                  Ask Questions. Rewrite Smart.
                </p>
                <p style={{ fontSize: 13, color: "#7c3aed", lineHeight: 1.6, margin: 0 }}>
                  Our AI asks the right questions about your resume and rewrites it with hallucination prevention — ensuring accuracy, clarity, and truthfulness.
                </p>
              </div>
              <div style={{
                width: 40, height: 40, background: "#ede9fe", borderRadius: 10,
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
              }}>
                <IconFile />
              </div>
            </div>
          </>
        )}

        {tab === "bias" && (
          <div style={{ padding: "28px 0" }}>
            <p style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", margin: "0 0 8px" }}>Bias Mirror</p>
            <p style={{ fontSize: 13, color: "#64748b", margin: "0 0 24px", lineHeight: 1.6 }}>
              India-specific bias patterns detected in your resume.
            </p>
            {[
              { label: "Masculine language Bias", risk: "High", pct: 88, color: "#ef4444" },
              { label: "College tier inference", risk: "Medium", pct: 62, color: "#f59e0b" },
              { label: "Gender marker language", risk: "Low", pct: 28, color: "#22c55e" },
            ].map((item) => (
              <div key={item.label} style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 14, fontWeight: 600, color: "#334155" }}>{item.label}</span>
                  <span style={{ fontSize: 12, fontWeight: 700, color: item.color, textTransform: "uppercase" }}>{item.risk}</span>
                </div>
                <Bar pct={item.pct} color={item.color} />
              </div>
            ))}
          </div>
        )}

        {tab === "rewrite" && (
          <div style={{ padding: "28px 0" }}>
            <p style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", margin: "0 0 8px" }}>Rewrite Engine</p>
            <p style={{ fontSize: 13, color: "#64748b", margin: "0 0 20px", lineHeight: 1.6 }}>
              Clarifying Q&A gates every rewrite — metrics stay real and defensible.
            </p>
            {[
              "Quantified impact bullets added",
              "Bias-triggering phrases removed",
              "ATS keyword density improved",
              "Hallucination prevention active",
            ].map((item) => (
              <div key={item} style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "12px 16px", background: "#f0f9ff",
                borderRadius: 12, marginBottom: 10, border: "1px solid #bae6fd",
              }}>
                <IconCheck />
                <span style={{ fontSize: 14, color: "#0369a1", fontWeight: 500 }}>{item}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const features = [
  { icon: <IconDoc />, title: "ATS Scoring", desc: "See how systems score your resume" },
  { icon: <IconBolt />, title: "7–8 Seconds, What Recruiter Thinks", desc: "Simulated recruiter scan in real time" },
  { icon: <IconShield />, title: "10 Bias Patterns Detected", desc: "India-specific insights that matter" },
];

export function LandingPage() {
  return (
    <>
      <style>{`
        .landing-root {
          min-height: 100vh;
          background: #f1f5f9;
          font-family: Inter, system-ui, -apple-system, sans-serif;
          box-sizing: border-box;
        }
        .landing-root *, .landing-root *::before, .landing-root *::after {
          box-sizing: border-box;
        }
        .hero-grid {
          max-width: 1140px;
          margin: 0 auto;
          padding: 52px 32px 60px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 48px;
          align-items: stretch;
        }
        .feature-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin-top: 36px;
        }
        .how-section {
          max-width: 1140px;
          margin: 0 auto;
          padding: 0 32px 72px;
        }
        .how-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
          margin-top: 28px;
        }
        @media (max-width: 900px) {
          .hero-grid { grid-template-columns: 1fr !important; gap: 36px; }
          .feature-grid { grid-template-columns: 1fr !important; }
          .how-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div className="landing-root">
        <div className="hero-grid">

          {/* ── LEFT ── */}
          <div>
            <span style={{
              display: "inline-block",
              background: "#fff",
              border: "1px solid #bfdbfe",
              color: "#2563eb",
              fontSize: 12,
              fontWeight: 500,
              borderRadius: 999,
              padding: "5px 14px",
              marginBottom: 24,
            }}>
              Bias-Aware Resume Intelligence System
            </span>

            <h1 style={{
              fontSize: "clamp(30px, 3.6vw, 46px)",
              fontWeight: 800,
              lineHeight: 1.13,
              color: "#0f172a",
              margin: "0 0 18px",
              letterSpacing: "-0.02em",
            }}>
              MirrorCue AI helps candidates see what recruiters unconsciously decode.
            </h1>

            <p style={{
              fontSize: 15,
              color: "#475569",
              lineHeight: 1.7,
              margin: "0 0 32px",
              maxWidth: 480,
            }}>
              ATS scoring, India-specific Bias Mirror insights, and verified rewrites — all in one guided workflow.
            </p>

            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <Link to="/register" style={{
                background: "#2563eb",
                color: "#fff",
                fontWeight: 600,
                fontSize: 14.5,
                padding: "12px 26px",
                borderRadius: 12,
                textDecoration: "none",
                display: "inline-block",
                letterSpacing: "-0.01em",
              }}>
                Get Started Free
              </Link>
              <a href="#how-it-works" style={{
                background: "#fff",
                color: "#0f172a",
                fontWeight: 600,
                fontSize: 14.5,
                padding: "12px 26px",
                borderRadius: 12,
                textDecoration: "none",
                border: "1px solid #e2e8f0",
                display: "inline-block",
              }}>
                See How It Works
              </a>
            </div>

            {/* Feature cards */}
            <div className="feature-grid">
              {features.map((f) => (
                <div key={f.title} style={{
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 18,
                  padding: "18px 14px",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                }}>
                  <div style={{
                    width: 42, height: 42,
                    background: "#eff6ff",
                    borderRadius: 10,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    marginBottom: 12,
                  }}>
                    {f.icon}
                  </div>
                  <p style={{ fontSize: 13, fontWeight: 700, color: "#2563eb", margin: "0 0 5px", lineHeight: 1.3 }}>
                    {f.title}
                  </p>
                  <p style={{ fontSize: 12, color: "#64748b", margin: 0, lineHeight: 1.5 }}>{f.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ── RIGHT ── */}
          <div>
            <PreviewCard />
          </div>
        </div>

        {/* How it works */}
        <div id="how-it-works" className="how-section">
          <div style={{
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: 20,
            padding: "36px 36px",
            boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
          }}>
            <h2 style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", margin: 0 }}>
              How MirrorCue AI Works
            </h2>
            <div className="how-grid">
              {[
                { n: "1.", title: "Upload and analyze", desc: "PDF upload, JD paste, and live pipeline progress." },
                { n: "2.", title: "Surface hidden bias", desc: "Split-screen cards show what the candidate wrote versus what a recruiter may decode." },
                { n: "3.", title: "Rewrite responsibly", desc: "Clarifying questions gate the rewrite so metrics stay real and defensible." },
              ].map((item) => (
                <div key={item.title} style={{
                  background: "#f8fafc",
                  border: "1px solid #e2e8f0",
                  borderRadius: 14,
                  padding: "20px 18px",
                }}>
                  <p style={{ fontSize: 12, fontWeight: 700, color: "#3b82f6", margin: "0 0 6px" }}>{item.n}</p>
                  <h3 style={{ fontSize: 14, fontWeight: 700, color: "#0f172a", margin: "0 0 6px" }}>{item.title}</h3>
                  <p style={{ fontSize: 12.5, color: "#64748b", margin: 0, lineHeight: 1.6 }}>{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
