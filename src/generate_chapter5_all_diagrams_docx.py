from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw

from generate_chapter5_docx import (
    FIG_DIR,
    ROOT,
    SECTIONS,
    build_app,
    build_content_types,
    build_core,
    build_document_rels,
    build_root_rels,
    build_settings,
    build_styles,
    empty_paragraph,
    image_paragraph,
    paragraph,
    section_properties,
)
from generate_chapter5_rtf import (
    arrow,
    centered,
    create_future_roadmap,
    create_juvenile_workflow,
    create_system_architecture,
    load_font,
    rounded_box,
)


OUTPUT = ROOT / "CHAPTER5_CONCLUSION_FUTURE_SCOPE_ALL_DIAGRAMS.docx"


def create_overall_outcome(path: Path):
    img = Image.new("RGB", (1350, 850), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(22)
    small_font = load_font(20)

    centered(draw, (0, 20, 1350, 70), "Overall Outcome of the Proposed System", title_font, "#15314b")

    input_title = (40, 100, 300, 150)
    centered(draw, input_title, "Real-World Inputs", load_font(24, bold=True), "#1d3557")

    input_boxes = [
        ((40, 170, 300, 245), "CMFRI Landing Data"),
        ((40, 270, 300, 345), "NOAA SST Data"),
        ((40, 370, 300, 445), "FishBase Maturity Data"),
        ((40, 470, 300, 545), "Field / PFZ Observations"),
    ]
    for box, label in input_boxes:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    integration = (380, 200, 700, 300)
    features = (380, 360, 700, 450)
    rounded_box(draw, integration, "#eef6ea", "#2a9d8f")
    rounded_box(draw, features, "#eef6ea", "#2a9d8f")
    centered(draw, integration, "Data Integration Layer", load_font(24, bold=True))
    centered(draw, features, "Feature Engineering", load_font(24, bold=True))

    tasks = [
        ((790, 150, 1110, 235), "Fish Availability\nPrediction"),
        ((790, 280, 1110, 365), "Catch Quantity\nEstimation"),
        ((790, 410, 1110, 495), "Juvenile-Risk\nAssessment"),
        ((790, 560, 1110, 650), "Decision Support\nEngine"),
    ]
    for box, label in tasks:
        rounded_box(draw, box, "#fff4e6", "#e76f51" if "Decision" not in label else "#355070")
        centered(draw, box, label, text_font)

    advisory = (1180, 250, 1310, 610)
    rounded_box(draw, advisory, "#f8fbff", "#355070")
    centered(draw, (1185, 260, 1305, 315), "Fishing\nAdvisory", load_font(24, bold=True))
    outputs = [
        ((1195, 330, 1295, 390), "Availability\nStatus"),
        ((1195, 405, 1295, 465), "Expected\nCatch"),
        ((1195, 480, 1295, 540), "Juvenile\nWarning"),
        ((1195, 555, 1295, 615), "Safe-Zone\nRecommendation"),
    ]
    for box, label in outputs:
        rounded_box(draw, box, "#eef6ea", "#2a9d8f")
        centered(draw, box, label, small_font)

    arrow(draw, (300, 208), (380, 240))
    arrow(draw, (300, 307), (380, 247))
    arrow(draw, (300, 408), (380, 257))
    arrow(draw, (300, 507), (380, 267))
    arrow(draw, (540, 300), (540, 360))
    arrow(draw, (700, 405), (790, 192))
    arrow(draw, (700, 405), (790, 322))
    arrow(draw, (700, 405), (790, 452))
    arrow(draw, (950, 495), (950, 560))
    arrow(draw, (1110, 192), (1180, 360))
    arrow(draw, (1110, 322), (1180, 435))
    arrow(draw, (1110, 452), (1180, 510))
    arrow(draw, (1110, 605), (1180, 585))

    img.save(path, format="PNG")


def create_decision_logic(path: Path):
    img = Image.new("RGB", (1250, 900), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(22)
    small_font = load_font(19)

    centered(draw, (0, 20, 1250, 70), "Juvenile-Risk and Safe-Zone Decision Logic", title_font, "#15314b")

    user_box = (465, 95, 785, 175)
    rounded_box(draw, user_box, "#f8fbff", "#355070")
    centered(draw, user_box, "User Input", load_font(24, bold=True))

    diamond = [(470, 255), (625, 180), (780, 255), (625, 330)]
    draw.polygon(diamond, fill="#fff3e6", outline="#bc6c25", width=3)
    centered(draw, (505, 215, 745, 295), "Species Available?", text_font)

    exact = (120, 400, 455, 500)
    entered = (120, 540, 455, 645)
    fallback = (795, 465, 1130, 565)
    rounded_box(draw, exact, "#f8fbff", "#355070")
    rounded_box(draw, entered, "#f8fbff", "#355070")
    rounded_box(draw, fallback, "#f8fbff", "#355070")
    centered(draw, exact, "Yes", load_font(24, bold=True), "#2d6a4f")
    centered(draw, entered, "Observed Length and\nMaturity Length Entered", small_font)
    centered(draw, fallback, "Environmental Juvenile\nModel", text_font)

    formula = (120, 700, 455, 795)
    risk = (460, 700, 790, 795)
    rounded_box(draw, formula, "#eef6ea", "#2a9d8f")
    rounded_box(draw, risk, "#eef6ea", "#2a9d8f")
    centered(draw, formula, "Exact Rule:\nJR = 1 - Observed_Length /\nMaturity_Length", small_font)
    centered(draw, risk, "Juvenile Risk Level", load_font(24, bold=True))

    diamond2 = [(835, 735), (975, 665), (1115, 735), (975, 805)]
    draw.polygon(diamond2, fill="#fff3e6", outline="#bc6c25", width=3)
    centered(draw, (885, 700, 1065, 770), "Risk High?", text_font)

    reject_box = (815, 185, 1135, 275)
    accept_box = (815, 315, 1135, 405)
    suggest_box = (815, 560, 1135, 645)
    summary_box = (815, 830, 1135, 890)
    rounded_box(draw, reject_box, "#fff1f2", "#c1121f")
    rounded_box(draw, accept_box, "#ecfdf3", "#2d6a4f")
    rounded_box(draw, suggest_box, "#eef6ea", "#2a9d8f")
    rounded_box(draw, summary_box, "#f8fbff", "#355070")
    centered(draw, reject_box, "Reject Current Zone", text_font)
    centered(draw, accept_box, "Fishing Zone Acceptable", text_font)
    centered(draw, suggest_box, "Suggest Nearby\nLower-Risk Zone", text_font)
    centered(draw, summary_box, "Display Final Prediction\nSummary", text_font)

    draw.text((365, 240), "Yes", font=small_font, fill="#2d6a4f")
    draw.text((805, 240), "No", font=small_font, fill="#c1121f")
    draw.text((1080, 690), "Yes", font=small_font, fill="#c1121f")
    draw.text((1090, 760), "No", font=small_font, fill="#2d6a4f")

    arrow(draw, (625, 175), (625, 190))
    arrow(draw, (500, 290), (290, 400))
    arrow(draw, (780, 255), (815, 510))
    arrow(draw, (290, 500), (290, 540))
    arrow(draw, (290, 645), (290, 700))
    arrow(draw, (455, 748), (460, 748))
    arrow(draw, (790, 748), (835, 735))
    arrow(draw, (975, 665), (975, 405))
    arrow(draw, (975, 805), (975, 830))
    arrow(draw, (975, 735), (975, 645))

    img.save(path, format="PNG")


def create_validation_flow(path: Path):
    img = Image.new("RGB", (1350, 700), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(22)

    centered(draw, (0, 20, 1350, 70), "Validation and Performance Evaluation Flow", title_font, "#15314b")

    dataset = (80, 190, 320, 270)
    preprocess = (390, 190, 690, 270)
    split = (760, 190, 1000, 270)
    for box, label in [
        (dataset, "Integrated Dataset"),
        (preprocess, "Preprocessing and\nBalancing"),
        (split, "Train-Test Split"),
    ]:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    models = [
        ((120, 390, 360, 470), "Random Forest"),
        ((420, 390, 660, 470), "Boosting"),
        ((720, 390, 960, 470), "Hybrid Model"),
        ((1020, 390, 1260, 470), "Juvenile Model"),
    ]
    for box, label in models:
        rounded_box(draw, box, "#fff4e6", "#e76f51")
        centered(draw, box, label, text_font)

    metrics = [
        ((160, 560, 430, 635), "Availability Accuracy"),
        ((540, 560, 810, 635), "Quantity RMSE"),
        ((920, 560, 1190, 635), "Juvenile Accuracy and\nF1-Score"),
    ]
    for box, label in metrics:
        rounded_box(draw, box, "#eef6ea", "#2a9d8f")
        centered(draw, box, label, text_font)

    report = (1045, 165, 1285, 255)
    rounded_box(draw, report, "#f8fbff", "#355070")
    centered(draw, report, "Validation Report", text_font)

    arrow(draw, (320, 230), (390, 230))
    arrow(draw, (690, 230), (760, 230))
    arrow(draw, (880, 270), (240, 390))
    arrow(draw, (880, 270), (540, 390))
    arrow(draw, (880, 270), (840, 390))
    arrow(draw, (1000, 270), (1140, 390))
    arrow(draw, (240, 470), (300, 560))
    arrow(draw, (540, 470), (675, 560))
    arrow(draw, (840, 470), (675, 560))
    arrow(draw, (1140, 470), (1055, 560))
    arrow(draw, (430, 595), (1045, 225))
    arrow(draw, (810, 595), (1045, 225))
    arrow(draw, (1190, 595), (1160, 255))

    img.save(path, format="PNG")


def create_training_pipeline(path: Path):
    img = Image.new("RGB", (1350, 760), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(22)

    centered(draw, (0, 20, 1350, 70), "Model Training and Validation Pipeline", title_font, "#15314b")

    steps = [
        ((80, 145, 360, 225), "Integrated Dataset"),
        ((410, 145, 710, 225), "Cleaning and Balancing"),
        ((760, 145, 1080, 225), "Feature Selection and\nEngineering"),
        ((1110, 145, 1290, 225), "Train-Test Split"),
    ]
    for box, label in steps:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    models = [
        ((95, 380, 330, 460), "Random Forest Model"),
        ((395, 380, 630, 460), "Boosting Model"),
        ((695, 380, 930, 460), "Hybrid Model"),
        ((995, 380, 1230, 460), "Juvenile-Risk Model"),
    ]
    for box, label in models:
        rounded_box(draw, box, "#fff4e6", "#e76f51")
        centered(draw, box, label, text_font)

    results = [
        ((130, 585, 390, 660), "Availability Accuracy"),
        ((480, 585, 740, 660), "Catch Quantity RMSE"),
        ((830, 585, 1090, 660), "Juvenile Accuracy /\nF1-Score"),
        ((1110, 585, 1290, 660), "Performance\nComparison"),
    ]
    for box, label in results:
        rounded_box(draw, box, "#eef6ea", "#2a9d8f")
        centered(draw, box, label, text_font)

    arrow(draw, (360, 185), (410, 185))
    arrow(draw, (710, 185), (760, 185))
    arrow(draw, (1080, 185), (1110, 185))
    arrow(draw, (1200, 225), (210, 380))
    arrow(draw, (1200, 225), (510, 380))
    arrow(draw, (1200, 225), (810, 380))
    arrow(draw, (1200, 225), (1110, 380))
    arrow(draw, (210, 460), (260, 585))
    arrow(draw, (510, 460), (610, 585))
    arrow(draw, (810, 460), (960, 585))
    arrow(draw, (1110, 460), (1200, 585))

    img.save(path, format="PNG")


def create_future_scope_expansion(path: Path):
    img = Image.new("RGB", (1500, 420), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(20)

    centered(draw, (0, 20, 1500, 70), "Future Scope Expansion Framework", title_font, "#15314b")

    boxes = [
        ((20, 170, 200, 255), "Current Implemented\nPrototype"),
        ((230, 170, 435, 255), "Monthly and District-Level\nFisheries Data"),
        ((465, 170, 670, 255), "Additional Oceanographic\nVariables"),
        ((700, 170, 875, 255), "Native XGBoost\nDeployment"),
        ((905, 170, 1125, 255), "Larger Exact Juvenile\nObservation Dataset"),
        ((1155, 170, 1345, 255), "Mobile and Web-Based\nAdvisory System"),
        ((1375, 170, 1480, 255), "Large-Scale\nInstitutional\nValidation"),
    ]
    for box, label in boxes:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    points = [200, 435, 670, 875, 1125, 1345]
    next_points = [230, 465, 700, 905, 1155, 1375]
    for start, end in zip(points, next_points):
        arrow(draw, (start, 212), (end, 212))

    img.save(path, format="PNG")


def create_future_scope_detailed(path: Path):
    img = Image.new("RGB", (1500, 420), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(20)

    centered(draw, (0, 20, 1500, 70), "Future Scope Roadmap", title_font, "#15314b")

    boxes = [
        ((20, 170, 200, 255), "Current Working\nPrototype"),
        ((230, 170, 440, 255), "Larger Monthly /\nDistrict-Level Dataset"),
        ((470, 170, 670, 255), "More Oceanographic\nVariables"),
        ((700, 170, 880, 255), "Native XGBoost\nRuntime"),
        ((910, 170, 1130, 255), "More Exact Juvenile\nField Records"),
        ((1160, 170, 1360, 255), "Mobile / Web Advisory\nDeployment"),
        ((1390, 170, 1480, 255), "Government /\nField Validation"),
    ]
    for box, label in boxes:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    points = [200, 440, 670, 880, 1130, 1360]
    next_points = [230, 470, 700, 910, 1160, 1390]
    for start, end in zip(points, next_points):
        arrow(draw, (start, 212), (end, 212))

    img.save(path, format="PNG")


def ensure_diagrams():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    create_overall_outcome(FIG_DIR / "overall_outcome_diagram.png")
    create_decision_logic(FIG_DIR / "decision_logic_diagram.png")
    create_validation_flow(FIG_DIR / "validation_performance_flow.png")
    create_training_pipeline(FIG_DIR / "training_validation_pipeline.png")
    create_future_scope_detailed(FIG_DIR / "future_scope_detailed.png")
    create_future_scope_expansion(FIG_DIR / "future_scope_expansion.png")
    create_system_architecture(FIG_DIR / "system_architecture_embedded.png")
    create_juvenile_workflow(FIG_DIR / "juvenile_safe_zone_embedded.png")
    create_future_roadmap(FIG_DIR / "future_scope_roadmap_embedded.png")


FIGURES = [
    ("Figure 5.1 Overall architecture and final outcome of the proposed fish catch prediction and juvenile-risk assessment system.", FIG_DIR / "overall_outcome_diagram.png"),
    ("Figure 5.2 Block diagram of the proposed fish catch prediction and juvenile-risk assessment system.", FIG_DIR / "system_architecture_embedded.png"),
    ("Figure 5.3 Decision flow used for exact juvenile-risk assessment and safe-zone recommendation.", FIG_DIR / "decision_logic_diagram.png"),
    ("Figure 5.4 Exact and fallback juvenile-risk assessment workflow used in the proposed system.", FIG_DIR / "juvenile_safe_zone_embedded.png"),
    ("Figure 5.5 Validation workflow used to measure classification, regression, and juvenile-risk performance.", FIG_DIR / "validation_performance_flow.png"),
    ("Figure 5.6 Training, testing, and evaluation flow of the machine learning models.", FIG_DIR / "training_validation_pipeline.png"),
    ("Figure 5.7 Comparison of fish availability accuracy across the implemented models.", FIG_DIR / "availability_accuracy.png"),
    ("Figure 5.8 Comparison of catch quantity prediction error across the implemented models.", FIG_DIR / "quantity_rmse.png"),
    ("Figure 5.9 Confusion matrix of the juvenile-risk assessment model.", FIG_DIR / "juvenile_confusion_matrix.png"),
    ("Figure 5.10 Future enhancement roadmap for improving scale, accuracy, and practical deployment.", FIG_DIR / "future_scope_detailed.png"),
    ("Figure 5.11 Future enhancement roadmap for improving data scale, model performance, and field usability.", FIG_DIR / "future_scope_expansion.png"),
    ("Figure 5.12 Performance comparison before and after improvement of the implemented models.", FIG_DIR / "improvement_comparison.png"),
]


SECTION_FIGURES = {
    "5.1 Conclusion": [0, 1],
    "5.2 Contributions of the Proposed System": [2, 3],
    "5.3 Practical Implications and Applications": [4, 5, 6, 7, 8],
    "5.4 Recommendations for Future Work": [9, 10, 11],
}


def build_document_xml():
    body = []
    relationships: list[tuple[str, Path]] = []
    docpr_id = 1

    body.append(paragraph("Chapter 5 Conclusion and Future Scope of Work", center=True, bold=True, size_half_points=36, after=220))

    for heading_text, paragraphs in SECTIONS.items():
        body.append(paragraph(heading_text, bold=True, size_half_points=30, after=160))
        for item in paragraphs:
            body.append(paragraph(item))

        for figure_pos in SECTION_FIGURES.get(heading_text, []):
            caption, path = FIGURES[figure_pos]
            rid = f"rId{len(relationships) + 1}"
            relationships.append((rid, path))
            body.append(paragraph(caption, center=True, italic=True, size_half_points=22, after=90))
            body.append(image_paragraph(rid, docpr_id, path))
            body.append(empty_paragraph())
            docpr_id += 1

    body.append(section_properties())
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
  xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
  xmlns:o="urn:schemas-microsoft-com:office:office"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
  xmlns:v="urn:schemas-microsoft-com:vml"
  xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:w10="urn:schemas-microsoft-com:office:word"
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
  xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
  xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
  xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
  xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
  mc:Ignorable="w14 wp14">
  <w:body>
    {''.join(body)}
  </w:body>
</w:document>
"""
    return xml, relationships


def main():
    ensure_diagrams()
    document_xml, relationships = build_document_xml()
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", build_content_types(relationships))
        docx.writestr("_rels/.rels", build_root_rels())
        docx.writestr("docProps/core.xml", build_core())
        docx.writestr("docProps/app.xml", build_app())
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/_rels/document.xml.rels", build_document_rels(relationships))
        docx.writestr("word/styles.xml", build_styles())
        docx.writestr("word/settings.xml", build_settings())
        for _, path in relationships:
            docx.write(path, f"word/media/{path.name}")
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
