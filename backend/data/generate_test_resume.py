from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_demo_resume(output_path: str = "backend/data/demo_resume.pdf") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    story = [
        Paragraph("Aarav Kumar", styles["Title"]),
        Paragraph("aarav.kumar@example.com | Chennai | +91 9876543210", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Education", styles["Heading2"]),
        Paragraph("Panimalar Engineering College — B.Tech AI & DS — CGPA 6.8/10", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Experience", styles["Heading2"]),
        Paragraph("ML Intern, Insight Labs — Worked on model training and reporting dashboards.", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Projects", styles["Heading2"]),
        Paragraph("Resume Ranker — Built a Python project for resume analysis using NLP and scoring logic.", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Skills", styles["Heading2"]),
        Paragraph("Python, SQL, Pandas, Scikit-learn, Power BI", styles["Normal"]),
    ]

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build(story)
    return str(path)


if __name__ == "__main__":
    generated = build_demo_resume()
    print(f"Generated test resume at {generated}")
