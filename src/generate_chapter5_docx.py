from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
OUTPUT = ROOT / "CHAPTER5_CONCLUSION_FUTURE_SCOPE_SUBMISSION.docx"


SECTIONS = {
    "5.1 Conclusion": [
        "The present project, titled Fish Catch Prediction and Juvenile Risk Assessment System, was developed as a practical and sustainability-oriented decision support framework for fishing operations. The implemented system combines fisheries landing information, sea surface temperature, maturity-based biological inputs, and field observations to predict fish availability, estimate catch quantity, assess juvenile risk, and recommend safer alternative fishing zones.",
        "A major strength of the proposed work is that it does not stop with a single prediction output. Instead, it integrates classification, regression, risk assessment, and recommendation in one workflow. The system predicts whether fishing conditions are favorable, estimates the likely catch quantity, checks whether the fish population is dominated by juvenile fish, and guides the user toward safer nearby zones when the risk is high.",
        "The project also demonstrates multiple machine learning pipelines. Fish availability is treated as a classification problem, catch quantity is modeled as a regression problem, and juvenile risk is assessed through both an exact maturity-based rule and an environmental fallback model. In the latest validation, the Boosting model produced the best availability accuracy of 68.18 percent, the Random Forest regressor achieved the lowest quantity RMSE of 37039.02, and the juvenile-risk model reached 77.27 percent accuracy with a weighted F1-score of 0.7538.",
        "Another important contribution is the maturity-based exact juvenile rule. Whenever species name, observed fish length, and maturity length are available, the system applies a biologically meaningful formula to estimate juvenile risk directly. When such values are not available, the environmental fallback layer is used so that the application remains functional. Therefore, the project demonstrates an executable and academically valid tabular machine learning system that is suitable for final-year project presentation and future extension.",
    ],
    "5.2 Contributions of the Proposed System": [
        "The proposed system makes both technical and practical contributions in the area of fisheries decision support. First, it integrates fish availability prediction, catch quantity estimation, juvenile-risk assessment, and safe-zone recommendation within a single working application. This makes the system more useful and more realistic than a project that focuses on only one output.",
        "Second, the project uses domain-relevant real-world sources such as CMFRI landing data, NOAA sea surface temperature records, FishBase maturity information, and PFZ-style field observations. Third, it implements multiple algorithmic paths including Random Forest, Boosting, Hybrid modeling, and a separate juvenile-risk pipeline. This enables comparative analysis and justified model selection.",
        "The fourth contribution is the use of exact maturity-based juvenile logic, which improves the biological authenticity of the system. The fifth contribution is the safe-zone advisory mechanism, which transforms the model outputs into a practical recommendation. Finally, the project includes testing, validation, and algorithm demonstration support, making it suitable for viva presentation and technical review.",
    ],
    "5.3 Practical Implications and Applications": [
        "The proposed system has practical significance in fisheries planning. It can support fishers in identifying favorable zones by estimating whether a given region is likely to provide usable catch. Such guidance can reduce unnecessary travel, save fuel, and improve decision making under uncertain marine conditions.",
        "The juvenile-risk module is particularly valuable from a sustainability perspective. Catching juvenile fish on a large scale affects stock replenishment and future fishery productivity. By comparing observed fish length against maturity length, the system introduces a biologically meaningful safeguard and supports responsible fishing practices.",
        "The catch quantity estimation component can also support storage planning, labor allocation, and local market coordination. In addition, the project can serve as a reference implementation for marine analytics, applied machine learning, and environmental decision support. Its modular design makes it suitable for future transformation into a broader web or mobile advisory platform.",
    ],
    "5.4 Recommendations for Future Work": [
        "Although the project has achieved its goal as a working prototype, several extensions can improve its practical strength. The first requirement is a larger dataset with monthly, district-level, harbor-level, or fishing-zone-level records. A richer dataset would allow the models to capture seasonal variation and local patterns more accurately.",
        "The second important direction is the addition of more oceanographic variables such as salinity, chlorophyll concentration, currents, mixed layer depth, and improved dissolved oxygen measurements. These variables are ecologically relevant and may improve both classification and regression performance.",
        "Further work should also enable native XGBoost execution and wider hyperparameter tuning. Another major improvement is the collection of more exact juvenile field observations, which would make the juvenile layer more statistically strong and biologically grounded. Future versions may additionally include uncertainty reporting, model explainability, GPS-assisted mobile access, and long-term validation against official advisories and fisheries records.",
        "Overall, the current work provides a strong academic prototype with clear scope for data expansion, model refinement, and field deployment. Its modular architecture, real-world orientation, and sustainability focus make it a suitable base for future research and practical fisheries applications.",
    ],
}


FIGURES = [
    (
        "Figure 5.1 Block diagram of the proposed fish catch prediction and juvenile-risk assessment system.",
        FIG_DIR / "system_architecture_embedded.png",
    ),
    (
        "Figure 5.2 Exact and fallback juvenile-risk assessment workflow with safe-zone recommendation.",
        FIG_DIR / "juvenile_safe_zone_embedded.png",
    ),
    (
        "Figure 5.3 Comparison of fish availability accuracy across the implemented models.",
        FIG_DIR / "availability_accuracy.png",
    ),
    (
        "Figure 5.4 Comparison of catch quantity prediction error across the implemented models.",
        FIG_DIR / "quantity_rmse.png",
    ),
    (
        "Figure 5.5 Confusion matrix of the juvenile-risk assessment model.",
        FIG_DIR / "juvenile_confusion_matrix.png",
    ),
    (
        "Figure 5.6 Performance comparison before and after improvement of the implemented models.",
        FIG_DIR / "improvement_comparison.png",
    ),
    (
        "Figure 5.7 Future roadmap for improving data scale, algorithm quality, and field deployment.",
        FIG_DIR / "future_scope_roadmap_embedded.png",
    ),
]


def emu_dimensions(image_path: Path, max_width_inches: float = 6.35) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width_px, height_px = img.size
    width_emu = int(max_width_inches * 914400)
    height_emu = int(width_emu * height_px / max(width_px, 1))
    return width_emu, height_emu


def run(text: str, bold: bool = False, italic: bool = False, size_half_points: int = 24) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    props.append(f'<w:sz w:val="{size_half_points}"/>')
    props.append(f'<w:szCs w:val="{size_half_points}"/>')
    prop_xml = "<w:rPr>" + "".join(props) + "</w:rPr>"
    return f"<w:r>{prop_xml}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"


def paragraph(text: str, center: bool = False, bold: bool = False, italic: bool = False, size_half_points: int = 24, after: int = 140) -> str:
    jc = "<w:jc w:val=\"center\"/>" if center else "<w:jc w:val=\"both\"/>"
    ppr = f"<w:pPr>{jc}<w:spacing w:after=\"{after}\"/></w:pPr>"
    return f"<w:p>{ppr}{run(text, bold=bold, italic=italic, size_half_points=size_half_points)}</w:p>"


def empty_paragraph() -> str:
    return "<w:p/>"


def image_paragraph(rid: str, docpr_id: int, image_path: Path) -> str:
    cx, cy = emu_dimensions(image_path)
    name = escape(image_path.name)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/><w:spacing w:after="100"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0"
        xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
        xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{docpr_id}" name="{name}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{name}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rid}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
""".strip()


def section_properties() -> str:
    return """
<w:sectPr>
  <w:pgSz w:w="12240" w:h="15840"/>
  <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
</w:sectPr>
""".strip()


def build_document_xml() -> tuple[str, list[tuple[str, Path]]]:
    body = []
    relationships = []

    body.append(paragraph("Chapter 5 Conclusion and Future Scope of Work", center=True, bold=True, size_half_points=36, after=220))

    figure_index = 0
    docpr_id = 1
    figure_map = {
        "5.1 Conclusion": [0],
        "5.2 Contributions of the Proposed System": [1],
        "5.3 Practical Implications and Applications": [2, 3, 4],
        "5.4 Recommendations for Future Work": [5, 6],
    }

    for heading_text, paragraphs in SECTIONS.items():
        body.append(paragraph(heading_text, bold=True, size_half_points=30, after=160))
        for item in paragraphs:
            body.append(paragraph(item))

        for figure_pos in figure_map.get(heading_text, []):
            caption, path = FIGURES[figure_pos]
            rid = f"rId{len(relationships) + 1}"
            relationships.append((rid, path))
            body.append(paragraph(caption, center=True, italic=True, size_half_points=22, after=90))
            body.append(image_paragraph(rid, docpr_id, path))
            docpr_id += 1
            body.append(empty_paragraph())
            figure_index += 1

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


def build_document_rels(relationships: list[tuple[str, Path]]) -> str:
    items = []
    for index, (rid, path) in enumerate(relationships, start=1):
        items.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{escape(path.name)}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(items)}
</Relationships>
"""


def build_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def build_content_types(relationships: list[tuple[str, Path]]) -> str:
    defaults = """
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
"""
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
{defaults}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def build_styles() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def build_settings() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:zoom w:percent="100"/>
</w:settings>
"""


def build_core() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Chapter 5 Conclusion and Future Scope</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def build_app() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>
"""


def main():
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
