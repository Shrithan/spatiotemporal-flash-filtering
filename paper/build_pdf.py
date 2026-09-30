"""Export the v1.0 technical report as a reproducible six-page PDF.

The canonical source remains ``main.tex`` and is validated with the desktop
LaTeX compiler. This portable exporter exists because that compiler exposes a
preview but no file-export API in automated runs. Numeric tables are loaded
from the same canonical CSV files used by the LaTeX report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "experiments/results/canonical"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=17, leading=20, alignment=TA_CENTER, spaceAfter=7),
        "author": ParagraphStyle("Author", parent=base["Normal"], fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=8),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12, leading=14, spaceBefore=7, spaceAfter=4, textColor=colors.HexColor("#111827")),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=9.5, leading=11, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1f2937")),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=8.1, leading=10.1, spaceAfter=4, alignment=0),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontSize=7.1, leading=8.6, spaceAfter=2),
        "abstract": ParagraphStyle("Abstract", parent=base["BodyText"], fontSize=7.8, leading=9.5, leftIndent=20, rightIndent=20, spaceAfter=5),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontSize=7, leading=8.4, alignment=TA_CENTER, textColor=colors.HexColor("#374151"), spaceBefore=2, spaceAfter=4),
    }


def _p(story, text: str, style) -> None:
    story.append(Paragraph(text, style))


def _table(data, widths, font_size=6.8):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.4),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#9ca3af")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


def _figure(story, path: Path, width: float, height: float, caption: str, styles) -> None:
    image = Image(str(path), width=width, height=height, kind="proportional")
    image.hAlign = "CENTER"
    story.extend([image, Paragraph(caption, styles["caption"])])


def _page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d1d5db"))
    canvas.line(.55 * inch, .48 * inch, 7.95 * inch, .48 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(.55 * inch, .31 * inch, "Spatiotemporal Detection and Localized Suppression of Rapid Visual Changes in Video")
    canvas.drawRightString(7.95 * inch, .31 * inch, str(doc.page))
    canvas.restoreState()


def build(output: Path = PAPER / "main.pdf") -> Path:
    styles = _styles()
    doc = BaseDocTemplate(str(output), pagesize=letter, leftMargin=.55*inch, rightMargin=.55*inch, topMargin=.48*inch, bottomMargin=.58*inch,
                          title="Spatiotemporal Detection and Localized Suppression of Rapid Visual Changes in Video",
                          author="Shrithan Adipuram")
    doc.addPageTemplates(PageTemplate(id="report", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")], onPage=_page))
    story = []
    main = pd.read_csv(RESULTS / "summary.csv").set_index("method")
    sweep = pd.read_csv(RESULTS / "sweep/parameter_sweep_summary.csv")
    ablation = pd.read_csv(RESULTS / "adaptive_ablation/adaptive_ablation_summary.csv").set_index("config_id")

    _p(story, "Spatiotemporal Detection and Localized Suppression<br/>of Rapid Visual Changes in Video", styles["title"])
    _p(story, "Shrithan Adipuram<br/>University of Wisconsin-Madison<br/>September 2026", styles["author"])
    _p(story, "Abstract", styles["h1"])
    _p(story, "We investigate whether spatially localized temporal filtering can reduce rapid visual changes while preserving more source information than whole-frame filtering. The artifact implements no-filter, global, and block-localized baselines plus an event-specific adaptive method with separate luminance-transition, saturated-red, and regular-pattern evidence channels. A deterministic 12-scenario synthetic benchmark, a compact parameter sweep, component ablations, and a natural-media case study provide evaluation. At default settings, adaptive filtering reduces aggregate mean activity from 0.10866 to 0.02966 and attains mask IoU 0.8907, but incurs MAE 0.05624 and leaves aggregate peak activity 0.16744. No adaptive configuration lies on the aggregate MAE-activity frontier. The lyric-video case study reduces mean activity from 0.06293 to 0.01408 while visibly altering the sequence and leaving peak 0.11285. These values are computational proxies, not clinical validation, seizure-risk estimates, or guarantees of standards compliance.", styles["abstract"])
    _p(story, "1. Introduction", styles["h1"])
    _p(story, "Rapid changes may cover a complete frame or a small object. A global spatial average can dilute a small intense event, while a whole-frame response can alter unrelated content. The research question is: at comparable temporal suppression, how do global, localized, and event-specific methods differ in preservation of the source video? The contribution is a reproducible, inspectable testbed with explicit color handling, controlled ground truth, per-case metrics, a parameter study, ablations, and executable tests. Reconstruction methods remain exploratory rather than central.", styles["body"])
    _p(story, "The project measures decoded-pixel signals only. Computational temporal activity is not clinical seizure risk. It does not model calibrated display luminance, viewing geometry, HDR, or individual physiology.", styles["body"])
    _p(story, "2. Related Work", styles["h1"])
    _p(story, "Clinical and psychophysical literature identifies flashes, long-wavelength red transitions, and high-contrast regular patterns as relevant stimulus classes [1-5]. Broadcast and web guidance operationalize parts of that evidence [12-15], but standards-inspired analysis is not formal conformance. Computational work includes adaptive temporal filtering [6], automatic and parallel detection [7,8], learned video transformation [9], and user-facing GIF defenses [10]. This project emphasizes transparent components, spatial ground truth, and suppression-distortion measurement; it does not claim novelty or duplicate a normative analyzer.", styles["body"])
    _p(story, "3. Problem Formulation", styles["h1"])
    _p(story, "Frames are decoded as normalized sRGB. The sRGB inverse transfer function is applied channel-wise, then relative linear luminance is Y = 0.2126 R_lin + 0.7152 G_lin + 0.0722 B_lin. sRGB and BT.709 share primaries and D65, hence the linear-light coefficients; BT.709 does not specify sRGB decoding. The optional encoded-RGB mode is explicitly called a brightness proxy.", styles["body"])
    _p(story, "Per-pixel activity is Delta_t(x,y) = |Y_t(x,y) - Y_(t-1)(x,y)| and the global score is D_t = mean(Delta_t). If fraction p changes by magnitude a, D_t = pa, demonstrating spatial dilution. Conceptually, the project minimizes source distortion D(V,V_hat) subject to computational activity R(V_hat) <= tau. Both objectives are reported.", styles["body"])
    story.append(PageBreak())
    _p(story, "4. Methods", styles["h1"])
    _p(story, "Baselines", styles["h2"])
    _p(story, "No filtering supplies the zero-distortion reference. Global filtering blends the entire frame toward the prior filtered output when D_t crosses a threshold. Localized filtering averages recent change maps, thresholds fixed block means, optionally closes gaps, and Gaussian-feathers the mask M_t. It computes I_hat_t = (1-alpha M_t) I_t + alpha M_t I_hat_(t-1). All operations are O(HW) per frame.", styles["body"])
    _p(story, "Event-Specific Adaptive Filtering", styles["h2"])
    _p(story, "Separate masks encode opposing luminance transitions, a documented saturated-red dominance proxy, and regular horizontal/vertical stripe profiles. Luminance evidence limits the current linear-luminance step around the previous output. Red evidence desaturates toward equal RGB channels at the same luminance. Pattern evidence contracts local contrast around a Gaussian mean. The masks are feathered and composed locally. Opposing-transition logic requires a reversal, so the first transition remains untreated.", styles["body"])
    _figure(story, PAPER / "figures/adaptive_architecture.png", 6.75*inch, 3.9*inch,
            "Figure 1. Implemented event-specific architecture. Each evidence channel maps to a distinct local correction.", styles)
    story.append(PageBreak())

    _p(story, "5. Experimental Setup", styles["h1"])
    _p(story, "Seed 7 produces twelve 24-frame, 96 x 64, 24-fps RGB sequences: whole-frame, small, medium, multiple, and rapid localized alternation; static texture; gradual illumination; scene cut; global exposure change; moving bright object; localized red alternation; and a persistent regular pattern. Exact masks are supplied where event localization is meaningful. Empty masks identify designed confounds, not medically safe content.", styles["body"])
    _p(story, "Metrics include mean/peak activity, threshold-exceeding frames, high-change area, residual activity in labeled regions, MAE, MSE, PSNR, SSIM, modified area, precision, recall, F1, IoU, time, and FPS. The sweep evaluates 49 operating points: global thresholds {.04,.08,.12,.18} and blends {.35,.65,.85}; localized thresholds {.06,.12,.18}, those blends, and block sizes {4,8}; adaptive luminance thresholds {.05,.10,.15}, maximum steps {.03,.08,.15}, and two strength profiles. Pareto efficiency minimizes aggregate MAE and mean activity within this finite grid.", styles["body"])
    _p(story, "The natural-media case study uses a 47-second, 1280 x 720, 23.976-fps excerpt from the official BREAK MY SOUL lyric video. It is a natural-media demonstration without a clinical label; source media is not distributed.", styles["body"])
    _p(story, "6. Synthetic Results", styles["h1"])
    rows = [["Method", "Activity", "Peak", "MAE", "SSIM", "Area", "IoU"]]
    labels = [("none","None"),("global","Global"),("localized","Localized"),("adaptive","Adaptive"),("enhanced_exploratory","Enhanced*"),("framegen_exploratory","Framegen*")]
    for key, label in labels:
        row = main.loc[key]
        rows.append([label, f"{row.mean_activity:.4f}", f"{row.peak_activity:.4f}", f"{row.mae:.4f}", f"{row.ssim:.3f}", f"{row.modified_area_ratio:.3f}", f"{row.iou:.3f}"])
    story.append(_table(rows, [1.1*inch]+[.73*inch]*6))
    _p(story, "Table 1. Canonical twelve-case means. *Exploratory methods. Lower is better for activity, peak, MAE, and area; higher is better for SSIM and IoU.", styles["caption"])
    _p(story, "Adaptive filtering has the highest default IoU and lower mean activity than the central baselines, but the greatest default MAE and a peak close to no filtering. Global filtering has the best default SSIM. On the small square, global filtering does nothing; localized filtering lowers mean activity from .01007 to .00198 while modifying about 1.0% of locations. For whole-frame alternation, global and localized outputs are identical.", styles["body"])
    _figure(story, RESULTS / "sweep/suppression_distortion_pareto.png", 6.45*inch, 4.3*inch,
            "Figure 2. All 49 operating points and the 17-point non-dominated MAE-activity frontier.", styles)
    story.append(PageBreak())

    _p(story, "6.1 Suppression-Distortion Analysis", styles["h1"])
    best = sweep.loc[sweep.groupby("method").mean_activity.idxmin()].set_index("method")
    _p(story, f"None of the 18 adaptive configurations is non-dominated. Its strongest evaluated setting reaches activity {best.loc['adaptive','mean_activity']:.5f} at MAE {best.loc['adaptive','mae']:.5f}; the strongest localized setting reaches lower activity {best.loc['localized','mean_activity']:.5f} at lower MAE {best.loc['localized','mae']:.5f}. This rejects a claim that the current adaptive method supplies the best aggregate two-objective tradeoff. It does not establish universal inferiority under other metrics or data.", styles["body"])
    _p(story, "6.2 Natural-Media Case Study", styles["h1"])
    _p(story, "Mean activity in the lyric excerpt falls from .06293 to .01408 (77.6%). At the strongest source transition it falls from .11880 to .02770, but the maximum elsewhere in the filtered excerpt is .11285. Mean absolute RGB modification is .08955 and 59.6% of decoded pixel locations change by more than one 8-bit code value. The actual frame pairs below expose the visible black-to-gray alteration; suppression is not free.", styles["body"])
    _figure(story, PAPER / "figures/lyric_adaptive/frame_sequence_comparison.png", 7.05*inch, 5.2*inch,
            "Figure 3. Actual adjacent source and adaptive frames at the three strongest source transitions. Times are relative to the extracted segment.", styles)
    story.append(PageBreak())

    _p(story, "7. Adaptive Component Ablation", styles["h1"])
    order=["luminance_only","red_only","pattern_only","luminance_red","luminance_pattern","red_pattern","all_channels"]
    labels={"luminance_only":"L","red_only":"R","pattern_only":"P","luminance_red":"L+R","luminance_pattern":"L+P","red_pattern":"R+P","all_channels":"L+R+P"}
    rows=[["Channels","Activity","Peak","MAE","Area","IoU"]]
    for key in order:
        row=ablation.loc[key]; rows.append([labels[key],f"{row.mean_activity:.5f}",f"{row.peak_activity:.5f}",f"{row.mae:.5f}",f"{row.modified_area_ratio:.4f}",f"{row.iou:.3f}"])
    story.append(_table(rows,[1.0*inch]+[.86*inch]*5))
    _p(story, "Table 2. L=luminance, R=red, P=pattern. Values are twelve-case means.", styles["caption"])
    _p(story, "The luminance branch supplies nearly all aggregate activity suppression. Red evidence changes mean activity from .02985 to .02966 when added to luminance but increases MAE. Pattern correction changes a static pattern and increases MAE without reducing temporal activity. All channels improve synthetic-mask IoU because the suite includes labeled red and pattern cases, yet the complete configuration is not the best preservation choice.", styles["body"])
    _figure(story, PAPER / "figures/adaptive_ablation.png", 6.35*inch, 2.65*inch,
            "Figure 4. Suppression and distortion generated from the adaptive ablation CSV.", styles)
    _p(story, "8. Failure Cases and Limitations", styles["h1"])
    _p(story, "Global and localized baselines respond to an isolated scene cut, modifying 4.17% and 8.33% of locations despite an empty event mask. Localized filtering modifies 2.11% on a moving bright object. No default method responds to the designed global exposure change. For whole-frame alternation, adaptive peak remains .75431 despite reducing mean activity from .75431 to .10932 because the first transition precedes reversal evidence.", styles["body"])
    _figure(story, PAPER / "figures/failure_cases.png", 6.55*inch, 2.62*inch,
            "Figure 5. Measured behavior on scene-cut, motion, and exposure confounds.", styles)
    story.append(PageBreak())

    _p(story, "8. Failure Cases and Limitations (continued)", styles["h1"])
    _p(story, "Raw differences cannot distinguish object motion or a cut from a relevant intensity event. Recursive temporal blending can ghost moving content, while reconstruction can introduce interpolation artifacts. Pattern recognition is restricted to horizontal and vertical tile profiles, and its correction may alter static appearance without temporal benefit. The red ratio is not perceptually uniform. Compression changes decoded pixels and contaminates modification counts.", styles["body"])
    _p(story, "The dataset is small, synthetic, SDR, and internally uncompressed. Its masks represent design intent, not medical outcome. The real-media example has no clinical annotation. The implementation omits calibrated display output, viewing geometry, HDR, physiology, complete standards decision logic, and formal validation against independent commercial analyzers. Runtime is hardware-dependent. Neither a threshold nor a filtered output should be described as medically safe.", styles["body"])
    _p(story, "9. Conclusion", styles["h1"])
    _p(story, "Localized filtering avoids whole-frame modification on small controlled regions, but the best method depends on suppression level and distortion budget. The event-specific method improves interpretability and localization, yet is not Pareto-superior on aggregate MAE and mean activity. Its luminance branch supplies almost all measured suppression; red and pattern branches add scenario-specific behavior and distortion. The real-media case confirms substantial mean reduction alongside visible alteration and a remaining large peak. This artifact establishes a reproducible framework and falsifiable findings, not a finished safety system. Scene-cut rejection, motion compensation, local temporal-frequency estimation, stronger chromatic modeling, and held-out parameter selection are the most defensible next experiments.", styles["body"])
    _p(story, "References", styles["h1"])
    refs = [
        "[1] R. Fisher et al. Visually sensitive seizures: An updated review. Epilepsia 63(4), 2022. doi:10.1111/epi.17175.",
        "[2] R. Fisher et al. Photic- and pattern-induced seizures: a review. Epilepsia 46(9), 2005. doi:10.1111/j.1528-1167.2005.31405.x.",
        "[3] G. Harding and P. Harding. Photosensitive epilepsy and image safety. Applied Ergonomics 41(4), 2010. doi:10.1016/j.apergo.2008.08.005.",
        "[4] A. Wilkins, J. Emmett, G. Harding. Characterizing patterned images that precipitate seizures. Epilepsia 46(8), 2005. doi:10.1111/j.1528-1167.2005.01405.x.",
        "[5] V. Porciatti et al. Lack of cortical contrast gain control in human photosensitive epilepsy. Nature Neuroscience 3, 2000. doi:10.1038/72972.",
        "[6] M. Nomura et al. A new adaptive temporal filter. Psychiatry and Clinical Neurosciences 54(6), 2000. doi:10.1046/j.1440-1819.2000.00770.x.",
        "[7] L. Carreira et al. Automatic detection of flashing video content. QoMEX, 2015. doi:10.1109/QoMEX.2015.7148104.",
        "[8] M. Alzubaidi et al. Parallel scheme for real-time detection. Computers in Biology and Medicine 70, 2016. doi:10.1016/j.compbiomed.2016.01.008.",
        "[9] A. Barbu, D. Banda, B. Katz. Deep video-to-video transformations for accessibility. Pattern Recognition Letters 137, 2020. doi:10.1016/j.patrec.2019.01.019.",
        "[10] L. South, D. Saffo, M. Borkin. Detecting and defending against seizure-inducing GIFs. CHI, 2021. doi:10.1145/3411764.3445510.",
        "[11] J. Jordan and G. Vanderheiden. International guidelines for photosensitive epilepsy. ACM TACCESS 17(3), 2024. doi:10.1145/3694790.",
        "[12] ITU-R BT.1702-3, Guidance for the reduction of photosensitive epileptic seizures caused by television, 2023.",
        "[13] ITU-R BT.709-6, HDTV parameter values, 2015.",
        "[14] W3C. Understanding Success Criterion 2.3.1: Three Flashes or Below Threshold, 2023.",
        "[15] W3C. Relative Luminance Definition, 2023.",
        "[16] Beyonce. BREAK MY SOUL (Official Lyric Video), YouTube, 2022.",
    ]
    for reference in refs:
        _p(story, reference, styles["small"])
    _p(story, "Source of record: paper/main.tex. Numeric source: experiments/results/canonical/. This PDF was generated by paper/build_pdf.py after the LaTeX source compiled successfully in the desktop editor.", styles["small"])

    doc.build(story)
    return output


if __name__ == "__main__":
    print(build())
