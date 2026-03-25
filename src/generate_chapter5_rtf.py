from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
OUTPUT_RTF = ROOT / "CHAPTER5_CONCLUSION_FUTURE_SCOPE_FULL.rtf"


TITLE = "Chapter 5 Conclusion and Future Scope of Work"

SECTION_TEXT = {
    "5.1 Conclusion": [
        "The present project, titled Fish Catch Prediction and Juvenile Risk Assessment System, was developed as a practical and sustainability-oriented decision support framework for fishing operations. The implemented system combines fisheries landing information, sea surface temperature, maturity-based biological inputs, and field observations to predict fish availability, estimate catch quantity, assess juvenile risk, and recommend safer alternative zones.",
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


@dataclass
class FigureBlock:
    title: str
    image_path: Path


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
                "/Library/Fonts/Times New Roman Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/Library/Fonts/Times New Roman.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill="#1a1a1a"):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=4)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=4)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill="#444444", width=4):
    draw.line([start, end], fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    mag = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / mag, dy / mag
    px, py = -uy, ux
    head = 14
    left = (end[0] - ux * head - px * head * 0.6, end[1] - uy * head - py * head * 0.6)
    right = (end[0] - ux * head + px * head * 0.6, end[1] - uy * head + py * head * 0.6)
    draw.polygon([end, left, right], fill=fill)


def rounded_box(draw: ImageDraw.ImageDraw, box, fill, outline):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)


def create_system_architecture(path: Path):
    img = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(28, bold=True)
    text_font = load_font(22)
    small_font = load_font(20)

    centered(draw, (0, 15, 1200, 60), "Proposed Fish Catch Prediction System Architecture", title_font, "#1d3557")

    source_boxes = [
        ((40, 110, 290, 195), "CMFRI Landing Data"),
        ((40, 225, 290, 310), "NOAA SST Data"),
        ((40, 340, 290, 425), "FishBase Maturity Data"),
        ((40, 455, 290, 540), "PFZ / Field Observations"),
    ]
    for box, label in source_boxes:
        rounded_box(draw, box, "#f8fbff", "#264653")
        centered(draw, box, label, text_font)

    integration = (390, 220, 710, 340)
    features = (390, 390, 710, 490)
    rounded_box(draw, integration, "#eef6ea", "#2a9d8f")
    rounded_box(draw, features, "#eef6ea", "#2a9d8f")
    centered(draw, integration, "Data Integration and\nPreprocessing", text_font)
    centered(draw, features, "Feature Engineering", text_font)

    availability = (810, 120, 1110, 205)
    quantity = (810, 245, 1110, 330)
    juvenile = (810, 370, 1110, 465)
    advisory = (810, 525, 1110, 645)
    for box in [availability, quantity, juvenile]:
        rounded_box(draw, box, "#fff4e6", "#e76f51")
    rounded_box(draw, advisory, "#f8fbff", "#264653")
    centered(draw, availability, "Availability Model", text_font)
    centered(draw, quantity, "Quantity Model", text_font)
    centered(draw, juvenile, "Juvenile-Risk Layer\n(Exact + Fallback)", small_font)
    centered(draw, advisory, "Decision Support\nSafe-Zone Recommendation", small_font)

    arrow(draw, (290, 152), (390, 245))
    arrow(draw, (290, 267), (390, 267))
    arrow(draw, (290, 382), (390, 295))
    arrow(draw, (290, 497), (390, 315))
    arrow(draw, (550, 340), (550, 390))
    arrow(draw, (710, 440), (810, 162))
    arrow(draw, (710, 440), (810, 287))
    arrow(draw, (710, 440), (810, 417))
    arrow(draw, (960, 205), (960, 525))
    arrow(draw, (960, 330), (960, 525))
    arrow(draw, (960, 465), (960, 525))

    img.save(path, format="PNG")


def create_juvenile_workflow(path: Path):
    img = Image.new("RGB", (1100, 820), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(28, bold=True)
    text_font = load_font(22)
    small_font = load_font(19)

    centered(draw, (0, 18, 1100, 60), "Juvenile-Risk Assessment and Safe-Zone Recommendation", title_font, "#1d3557")

    input_box = (390, 90, 710, 175)
    rounded_box(draw, input_box, "#f8fbff", "#355070")
    centered(draw, input_box, "User Input", text_font)

    diamond = [(390, 245), (550, 175), (710, 245), (550, 315)]
    draw.polygon(diamond, fill="#fff3e6", outline="#bc6c25", width=3)
    centered(draw, (410, 205, 690, 285), "Species and\nLength Available?", text_font)

    exact = (110, 405, 430, 500)
    fallback = (670, 405, 990, 500)
    rounded_box(draw, exact, "#f8fbff", "#355070")
    rounded_box(draw, fallback, "#f8fbff", "#355070")
    centered(draw, exact, "Exact Maturity Rule\nJR = 1 - Lobs / Lmat", small_font)
    centered(draw, fallback, "Environmental Juvenile\nFallback Model", small_font)

    risk = (390, 565, 710, 650)
    rounded_box(draw, risk, "#f8fbff", "#355070")
    centered(draw, risk, "Juvenile Risk Level\nHigh / Medium / Low", small_font)

    diamond2 = [(390, 720), (550, 655), (710, 720), (550, 785)]
    draw.polygon(diamond2, fill="#fff3e6", outline="#bc6c25", width=3)
    centered(draw, (440, 695, 660, 745), "Risk High?", text_font)

    avoid = (55, 680, 290, 765)
    safe = (810, 680, 1045, 765)
    search = (55, 80, 305, 165)
    rounded_box(draw, avoid, "#fff1f2", "#c1121f")
    rounded_box(draw, safe, "#ecfdf3", "#2d6a4f")
    rounded_box(draw, search, "#ecfdf3", "#2d6a4f")
    centered(draw, avoid, "Avoid Current Zone", text_font)
    centered(draw, safe, "Accept Zone", text_font)
    centered(draw, search, "Safe-Zone Search", text_font)

    arrow(draw, (550, 175), (550, 205))
    arrow(draw, (450, 280), (310, 405))
    arrow(draw, (650, 280), (790, 405))
    arrow(draw, (310, 500), (470, 565))
    arrow(draw, (790, 500), (630, 565))
    arrow(draw, (550, 650), (550, 680))
    arrow(draw, (395, 720), (290, 723))
    arrow(draw, (705, 720), (810, 723))
    arrow(draw, (172, 680), (180, 165))

    img.save(path, format="PNG")


def create_future_roadmap(path: Path):
    img = Image.new("RGB", (1200, 360), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(28, bold=True)
    text_font = load_font(21)
    centered(draw, (0, 20, 1200, 60), "Future Scope Roadmap", title_font, "#1d3557")

    boxes = [
        ((30, 135, 210, 225), "Current Working\nPrototype"),
        ((250, 135, 450, 225), "Monthly / District\nLevel Dataset"),
        ((490, 135, 690, 225), "More Oceanic\nVariables"),
        ((730, 135, 930, 225), "Native XGBoost\nExecution"),
        ((970, 135, 1170, 225), "Large-Scale Field\nValidation"),
    ]
    for box, label in boxes:
        rounded_box(draw, box, "#f8fbff", "#355070")
        centered(draw, box, label, text_font)

    arrow(draw, (210, 180), (250, 180))
    arrow(draw, (450, 180), (490, 180))
    arrow(draw, (690, 180), (730, 180))
    arrow(draw, (930, 180), (970, 180))

    img.save(path, format="PNG")


def png_to_rtf(image_path: Path, width_inches: float = 6.2) -> str:
    with Image.open(image_path) as img:
        width_px, height_px = img.size
        width_twips = int(width_inches * 1440)
        height_twips = int(width_twips * height_px / max(width_px, 1))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        hex_data = buffer.getvalue().hex().upper()
    return (
        "{\\pard\\qc{\\pict\\pngblip"
        f"\\picw{width_px}\\pich{height_px}\\picwgoal{width_twips}\\pichgoal{height_twips}\n"
        + hex_data
        + "}\\par}\n"
    )


def rtf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def paragraph(text: str) -> str:
    wrapped = textwrap.fill(text, width=108)
    return f"\\pard\\qj\\sa140 {rtf_escape(wrapped)}\\par\n"


def heading(text: str, level: int = 2) -> str:
    size = 34 if level == 1 else 28
    return f"\\pard\\sb240\\sa120\\b\\fs{size} {rtf_escape(text)}\\b0\\fs24\\par\n"


def caption(text: str) -> str:
    return f"\\pard\\qc\\i\\sa160 {rtf_escape(text)}\\i0\\par\n"


def build_document():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    system_png = FIG_DIR / "system_architecture_embedded.png"
    juvenile_png = FIG_DIR / "juvenile_safe_zone_embedded.png"
    roadmap_png = FIG_DIR / "future_scope_roadmap_embedded.png"

    create_system_architecture(system_png)
    create_juvenile_workflow(juvenile_png)
    create_future_roadmap(roadmap_png)

    figures = {
        "Figure 5.1 Block diagram of the proposed fish catch prediction and juvenile-risk assessment system.": system_png,
        "Figure 5.2 Exact and fallback juvenile-risk assessment workflow with safe-zone recommendation.": juvenile_png,
        "Figure 5.3 Comparison of fish availability accuracy across the implemented models.": FIG_DIR / "availability_accuracy.png",
        "Figure 5.4 Comparison of catch quantity prediction error across the implemented models.": FIG_DIR / "quantity_rmse.png",
        "Figure 5.5 Confusion matrix of the juvenile-risk assessment model.": FIG_DIR / "juvenile_confusion_matrix.png",
        "Figure 5.6 Performance comparison before and after improvement of the implemented models.": FIG_DIR / "improvement_comparison.png",
        "Figure 5.7 Future roadmap for improving data scale, algorithm quality, and field deployment.": roadmap_png,
    }

    parts = [
        "{\\rtf1\\ansi\\deff0",
        "{\\fonttbl{\\f0 Times New Roman;}}",
        "\\paperw12240\\paperh15840\\margl1440\\margr1440\\margt1440\\margb1440",
        "\\fs24",
        "\\pard\\qc\\b\\fs40 " + rtf_escape(TITLE) + "\\b0\\fs24\\par\n",
    ]

    section_names = list(SECTION_TEXT.keys())
    for index, section in enumerate(section_names):
        parts.append(heading(section))
        for para in SECTION_TEXT[section]:
            parts.append(paragraph(para))

        if section == "5.1 Conclusion":
            cap = "Figure 5.1 Block diagram of the proposed fish catch prediction and juvenile-risk assessment system."
            parts.append(png_to_rtf(figures[cap]))
            parts.append(caption(cap))
        elif section == "5.2 Contributions of the Proposed System":
            cap = "Figure 5.2 Exact and fallback juvenile-risk assessment workflow with safe-zone recommendation."
            parts.append(png_to_rtf(figures[cap]))
            parts.append(caption(cap))
        elif section == "5.3 Practical Implications and Applications":
            for cap in [
                "Figure 5.3 Comparison of fish availability accuracy across the implemented models.",
                "Figure 5.4 Comparison of catch quantity prediction error across the implemented models.",
                "Figure 5.5 Confusion matrix of the juvenile-risk assessment model.",
            ]:
                parts.append(png_to_rtf(figures[cap]))
                parts.append(caption(cap))
        elif section == "5.4 Recommendations for Future Work":
            for cap in [
                "Figure 5.6 Performance comparison before and after improvement of the implemented models.",
                "Figure 5.7 Future roadmap for improving data scale, algorithm quality, and field deployment.",
            ]:
                parts.append(png_to_rtf(figures[cap]))
                parts.append(caption(cap))

        if index < len(section_names) - 1:
            parts.append("\\par\n")

    parts.append("}")
    OUTPUT_RTF.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    build_document()
    print(f"Generated {OUTPUT_RTF}")
