from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw

from generate_chapter5_docx import (
    FIG_DIR,
    ROOT,
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
from generate_chapter5_rtf import arrow, centered, load_font, rounded_box


OUTPUT = ROOT / "CHAPTER6_APPENDIX_WITH_IMAGES.docx"


REFERENCES = [
    "1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "2. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189-1232.",
    "3. Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely Randomized Trees. Machine Learning, 63, 3-42.",
    "4. CMFRI Fish Catch Estimates. https://www.cmfri.org.in/fish-catch-estimates",
    "5. CMFRI Methodology. https://www.cmfri.org.in/methodology",
    "6. NOAA Optimum Interpolation Sea Surface Temperature. https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
    "7. Copernicus Marine Physics Reanalysis. https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
    "8. Copernicus Ocean Colour Chlorophyll Product. https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_MY_009_103/description",
    "9. INCOIS PFZ Advisory. https://services.incois.gov.in/MarineFisheries/PfzAdvisory.action",
    "10. INCOIS Marine Fisheries Text Data. https://services.incois.gov.in/MarineFisheries/TextDataHome?mfid=1&request_locale=en",
    "11. FishBase Glossary: Length at First Maturity. https://www.fishbase.se/glossary/Glossary.php?q=length+at+first+maturity",
    "12. scikit-learn Documentation. https://scikit-learn.org/",
    "13. Streamlit Documentation. https://streamlit.io/",
]


APPENDIX_A = [
    "Appendix A summarizes the core algorithms implemented in the proposed system. Fish availability is predicted using classification models, catch quantity is estimated using regression models, and juvenile-risk is assessed through both exact biological logic and an environmental fallback model.",
    "A.1 Random Forest Classifier: The Random Forest availability model uses the engineered environmental and catch features to classify whether fishing conditions are favorable. It is robust to non-linear interactions and performs well on mixed tabular data.",
    "A.2 Boosting Model: The boosting path is designed for gradient boosting based learning. On the current Mac environment, the code executes a Gradient Boosting fallback when native XGBoost runtime support is not available.",
    "A.3 Hybrid Model: The hybrid model combines PCA-based dimensionality reduction with ensemble learners such as Random Forest, Extra Trees, and Boosting. This improves representation learning while preserving ensemble robustness.",
    "A.4 Juvenile-Risk Model: The juvenile-risk layer uses an Extra Trees classifier for the fallback environmental path and an exact maturity-based rule whenever species name, observed length, and maturity length are available.",
    "A.5 Exact Juvenile Formula: JR = 1 - (Observed Length / Maturity Length). If the observed length is well below maturity length, the zone is classified as high juvenile risk.",
    "A.6 Safe-Zone Recommendation: When a zone is not recommended, the decision engine searches nearby candidate coordinates and suggests an alternative lower-risk zone for safer fishing operations.",
]


APPENDIX_B = [
    "Appendix B records the main commands required to execute the project, rebuild the dataset, retrain the models, validate the pipeline, and run the user interface.",
    "$ .venv/bin/python src/check_external_datasets.py",
    "$ .venv/bin/python src/run_full_pipeline.py",
    "$ .venv/bin/python src/validate_project.py",
    "$ .venv/bin/python src/demo_algorithms.py",
    "$ .venv/bin/python src/generate_report_figures.py",
    "$ .venv/bin/python -m unittest discover -s tests",
    "$ .venv/bin/streamlit run src/app.py --server.headless true --server.port 8501",
    "The execution flow normally begins with data validation, continues with dataset rebuilding and model training, and then ends with report generation, automated tests, and application launch.",
]


APPENDIX_C = [
    "Appendix C explains how validation is carried out in the project. Validation is performed at three levels: data validation, model validation, and logic validation.",
    "C.1 Data Validation: Verify dataset source path, row count, required fields, field observation rows, and exact-ready juvenile rows.",
    "C.2 Model Validation: Measure fish availability accuracy, catch quantity RMSE, juvenile-risk accuracy, and weighted F1-score using the validation script.",
    "C.3 Logic Validation: Confirm that the exact juvenile formula behaves correctly when species and observed length are available, and verify the environmental fallback path when those values are absent.",
    "C.4 Current Validated Status: Dataset rows = 110, field observation rows = 4, exact-ready rows = 3, juvenile exact-label rows in training = 3, and automated tests passed = 9.",
    "C.5 Visual Validation: The attached figures show comparative accuracy, RMSE, confusion matrix, and improvement trends for the implemented models.",
]


def create_algorithm_overview(path: Path):
    img = Image.new("RGB", (1450, 860), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(22)
    small_font = load_font(19)

    centered(draw, (0, 18, 1450, 70), "Appendix A Key Algorithms Overview", title_font, "#15314b")

    input_box = (80, 260, 340, 350)
    rounded_box(draw, input_box, "#f8fbff", "#355070")
    centered(draw, input_box, "Integrated Feature Set", load_font(24, bold=True))

    alg_boxes = [
        ((450, 110, 740, 195), "Random Forest\nClassifier"),
        ((450, 230, 740, 315), "Boosting /\nGradient Boosting"),
        ((450, 350, 740, 435), "Hybrid PCA + RF +\nET + Boosting"),
        ((450, 470, 740, 555), "Extra Trees\nJuvenile Model"),
        ((450, 590, 740, 690), "Exact Maturity Rule\nJR = 1 - Lobs / Lmat"),
    ]
    for box, label in alg_boxes:
        rounded_box(draw, box, "#fff4e6", "#e76f51")
        centered(draw, box, label, text_font)

    output_boxes = [
        ((1040, 140, 1330, 220), "Fish Availability"),
        ((1040, 280, 1330, 360), "Catch Quantity"),
        ((1040, 420, 1330, 500), "Juvenile Risk"),
        ((1040, 560, 1330, 640), "Safe-Zone Advisory"),
    ]
    for box, label in output_boxes:
        rounded_box(draw, box, "#eef6ea", "#2a9d8f")
        centered(draw, box, label, text_font)

    fusion = (980, 690, 1380, 790)
    rounded_box(draw, fusion, "#f8fbff", "#355070")
    centered(draw, fusion, "Prediction Engine and Decision Fusion", load_font(24, bold=True))

    for y in [152, 272, 392, 512, 640]:
        arrow(draw, (340, 305), (450, y))
    arrow(draw, (740, 152), (1040, 180))
    arrow(draw, (740, 272), (1040, 320))
    arrow(draw, (740, 392), (1040, 460))
    arrow(draw, (740, 512), (1040, 460))
    arrow(draw, (740, 640), (1040, 600))
    for y in [180, 320, 460, 600]:
        arrow(draw, (1185, y + 40), (1185, 690))
    centered(draw, (60, 145, 360, 210), "Engineered Inputs", small_font, "#1d3557")
    centered(draw, (460, 55, 760, 95), "Model Layer", small_font, "#1d3557")
    centered(draw, (1050, 85, 1330, 120), "Task Outputs", small_font, "#1d3557")

    img.save(path, format="PNG")


def create_execution_workflow(path: Path):
    img = Image.new("RGB", (1500, 520), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(20)

    centered(draw, (0, 18, 1500, 70), "Appendix B Project Execution Workflow", title_font, "#15314b")

    steps = [
        ((20, 220, 200, 305), "Check External\nDatasets"),
        ((235, 220, 425, 305), "Run Full Pipeline"),
        ((460, 220, 650, 305), "Validate Project"),
        ((685, 220, 875, 305), "Generate Algorithm\nDemo"),
        ((910, 220, 1100, 305), "Run Unit Tests"),
        ((1135, 220, 1325, 305), "Launch Streamlit\nApplication"),
    ]
    for box, label in steps:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    for start, end in [(200, 235), (425, 460), (650, 685), (875, 910), (1100, 1135)]:
        arrow(draw, (start, 262), (end, 262))

    cmd_boxes = [
        ((15, 355, 205, 435), "check_external_\ndatasets.py"),
        ((230, 355, 430, 435), "run_full_\npipeline.py"),
        ((455, 355, 655, 435), "validate_project.py"),
        ((680, 355, 880, 435), "demo_algorithms.py"),
        ((905, 355, 1105, 435), "unittest discover"),
        ((1130, 355, 1330, 435), "streamlit run\nsrc/app.py"),
    ]
    for box, label in cmd_boxes:
        rounded_box(draw, box, "#eef6ea", "#2a9d8f")
        centered(draw, box, label, text_font)

    for x in [110, 330, 555, 780, 1005, 1230]:
        arrow(draw, (x, 305), (x, 355))

    img.save(path, format="PNG")


def create_validation_procedure(path: Path):
    img = Image.new("RGB", (1500, 780), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(30, bold=True)
    text_font = load_font(21)
    small_font = load_font(19)

    centered(draw, (0, 20, 1500, 70), "Appendix C Validation Procedure", title_font, "#15314b")

    start = (610, 90, 890, 170)
    rounded_box(draw, start, "#f8fbff", "#355070")
    centered(draw, start, "Integrated Dataset and\nTrained Models", text_font)

    data_val = (120, 270, 450, 365)
    model_val = (585, 270, 915, 365)
    logic_val = (1050, 270, 1380, 365)
    for box, label in [
        (data_val, "Data Validation"),
        (model_val, "Model Validation"),
        (logic_val, "Logic Validation"),
    ]:
        rounded_box(draw, box, "#fff4e6", "#e76f51")
        centered(draw, box, label, load_font(24, bold=True))

    data_checks = (70, 470, 500, 610)
    model_checks = (535, 470, 965, 610)
    logic_checks = (1000, 470, 1430, 610)
    rounded_box(draw, data_checks, "#eef6ea", "#2a9d8f")
    rounded_box(draw, model_checks, "#eef6ea", "#2a9d8f")
    rounded_box(draw, logic_checks, "#eef6ea", "#2a9d8f")
    centered(draw, data_checks, "Row count\nRequired columns\nField observations\nExact-ready entries", small_font)
    centered(draw, model_checks, "Availability accuracy\nQuantity RMSE\nJuvenile accuracy\nWeighted F1-score", small_font)
    centered(draw, logic_checks, "Exact maturity rule\nFallback behavior\nSafe-zone response\nRuntime execution", small_font)

    report = (610, 675, 890, 750)
    rounded_box(draw, report, "#f8fbff", "#355070")
    centered(draw, report, "Validation Report", load_font(24, bold=True))

    arrow(draw, (750, 170), (285, 270))
    arrow(draw, (750, 170), (750, 270))
    arrow(draw, (750, 170), (1215, 270))
    arrow(draw, (285, 365), (285, 470))
    arrow(draw, (750, 365), (750, 470))
    arrow(draw, (1215, 365), (1215, 470))
    arrow(draw, (285, 610), (610, 710))
    arrow(draw, (750, 610), (750, 675))
    arrow(draw, (1215, 610), (890, 710))

    img.save(path, format="PNG")


def ensure_diagrams():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    create_algorithm_overview(FIG_DIR / "appendix_algorithms_overview.png")
    create_execution_workflow(FIG_DIR / "appendix_execution_workflow.png")
    create_validation_procedure(FIG_DIR / "appendix_validation_procedure.png")


FIGURES = [
    ("Figure A.1 Core algorithm overview for the implemented fish catch prediction and juvenile-risk assessment system.", FIG_DIR / "appendix_algorithms_overview.png"),
    ("Figure B.1 Project execution workflow showing the main scripts and runtime stages.", FIG_DIR / "appendix_execution_workflow.png"),
    ("Figure C.1 Validation procedure used for data checks, model metrics, and logic verification.", FIG_DIR / "appendix_validation_procedure.png"),
    ("Figure C.2 Availability prediction accuracy of the implemented models.", FIG_DIR / "availability_accuracy.png"),
    ("Figure C.3 Catch quantity prediction RMSE comparison.", FIG_DIR / "quantity_rmse.png"),
    ("Figure C.4 Juvenile-risk confusion matrix after class balancing and model refinement.", FIG_DIR / "juvenile_confusion_matrix.png"),
    ("Figure C.5 Performance comparison before and after model improvement.", FIG_DIR / "improvement_comparison.png"),
]


def build_document_xml():
    body = []
    relationships: list[tuple[str, Path]] = []
    docpr_id = 1

    body.append(paragraph("Chapter 6 References", center=True, bold=True, size_half_points=36, after=220))
    body.append(paragraph("References", bold=True, size_half_points=30, after=150))
    for item in REFERENCES:
        body.append(paragraph(item))

    body.append(empty_paragraph())
    body.append(paragraph("Appendix", center=True, bold=True, size_half_points=34, after=220))

    body.append(paragraph("Appendix A: Key Algorithms", bold=True, size_half_points=30, after=150))
    for item in APPENDIX_A:
        body.append(paragraph(item))
    rid = f"rId{len(relationships) + 1}"
    relationships.append((rid, FIGURES[0][1]))
    body.append(paragraph(FIGURES[0][0], center=True, italic=True, size_half_points=22, after=90))
    body.append(image_paragraph(rid, docpr_id, FIGURES[0][1]))
    body.append(empty_paragraph())
    docpr_id += 1

    body.append(paragraph("Appendix B: Project Execution Commands", bold=True, size_half_points=30, after=150))
    for item in APPENDIX_B:
        size = 22 if item.startswith("$ ") else 24
        body.append(paragraph(item, size_half_points=size))
    rid = f"rId{len(relationships) + 1}"
    relationships.append((rid, FIGURES[1][1]))
    body.append(paragraph(FIGURES[1][0], center=True, italic=True, size_half_points=22, after=90))
    body.append(image_paragraph(rid, docpr_id, FIGURES[1][1]))
    body.append(empty_paragraph())
    docpr_id += 1

    body.append(paragraph("Appendix C: Validation Procedure", bold=True, size_half_points=30, after=150))
    for item in APPENDIX_C:
        body.append(paragraph(item))
    for caption, path in FIGURES[2:]:
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
