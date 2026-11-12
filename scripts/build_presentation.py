"""
Builds the complete professional 24-slide B.Tech Minor Project Presentation
for Attention U-Net Glacier Segmentation at NIT Raipur.
Saves to ~/Desktop/Glacier_Project_Presentation/ and mirror in scratch.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
OUTPUT_DIR_SCRATCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "presentation"))
OUTPUT_DIR_DESKTOP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "presentation"))
ASSETS_DIR = os.path.join(OUTPUT_DIR_SCRATCH, "figures")

os.makedirs(OUTPUT_DIR_SCRATCH, exist_ok=True)
os.makedirs(OUTPUT_DIR_DESKTOP, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Color Palette: Deep Navy & Glacier Cyan Professional Theme
COLOR_BG_DARK = RGBColor(11, 25, 44)        # #0B192C (Deep Space Navy)
COLOR_CARD_BG = RGBColor(30, 41, 59)        # #1E293B (Slate Navy Card)
COLOR_CARD_BORDER = RGBColor(56, 189, 248)  # #38BDF8 (Glacier Cyan)
COLOR_TEXT_WHITE = RGBColor(255, 255, 255)  # #FFFFFF
COLOR_TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8
COLOR_TEXT_BODY = RGBColor(226, 232, 240)   # #E2E8F0
COLOR_CYAN_ACCENT = RGBColor(56, 189, 248)  # #38BDF8
COLOR_GOLD_ACCENT = RGBColor(251, 191, 36)  # #FBBF24
COLOR_GREEN_ACCENT = RGBColor(52, 211, 153) # #34D399
COLOR_ROSE_ACCENT = RGBColor(244, 63, 94)   # #F43F5E
COLOR_FOOTER_TEXT = RGBColor(100, 116, 139) # #64748B


def set_slide_background(slide, prs):
    """Sets a solid deep navy background for the slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG_DARK


def add_slide_header(slide, section_tag: str, title: str, subtitle: str = ""):
    """Creates a consistent, professional header across all slides."""
    # Section Tag Pill
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(4.5), Inches(0.35))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = section_tag.upper()
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_CYAN_ACCENT

    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.65))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_WHITE

    # Subtitle (optional)
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(0.35))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.size = Pt(11.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED


def add_slide_footer(slide, current_slide: int, total_slides: int = 24):
    """Creates a consistent footer on every slide."""
    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.7), Inches(0.3))
    tf = footer_box.text_frame
    p = tf.paragraphs[0]
    p.text = f"Attention U-Net Glacier Segmentation  |  NIT Raipur Minor Project"
    p.font.size = Pt(9)
    p.font.color.rgb = COLOR_FOOTER_TEXT

    # Page number right-aligned
    p_num = tf.add_paragraph()
    p_num.text = f"{current_slide} / {total_slides}"
    p_num.font.size = Pt(9)
    p_num.font.bold = True
    p_num.font.color.rgb = COLOR_CYAN_ACCENT
    p_num.alignment = PP_ALIGN.RIGHT


def add_card(slide, left, top, width, height, border_color=COLOR_CARD_BORDER, bg_color=COLOR_CARD_BG):
    """Creates a stylized card container for content or images."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.2)
    return shape


def add_bullet_point(text_frame, bold_prefix: str, body_text: str, pt_size: int = 12, space_after: int = 10, prefix_color=COLOR_CYAN_ACCENT):
    """Helper to add clean bold-prefix bullet points."""
    p = text_frame.add_paragraph()
    p.space_after = Pt(space_after)
    p.font.size = Pt(pt_size)

    run_bold = p.add_run()
    run_bold.text = bold_prefix + " "
    run_bold.font.bold = True
    run_bold.font.color.rgb = prefix_color

    run_body = p.add_run()
    run_body.text = body_text
    run_body.font.color.rgb = COLOR_TEXT_BODY


def build_presentation():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    print("[*] Creating 24 Professional Research Presentation Slides...")

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, prs)

    # Accent decorative banner
    banner = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.12), Inches(4.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = COLOR_CYAN_ACCENT
    banner.line.fill.background()

    # Department tag
    tag_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(11.0), Inches(0.4))
    p = tag_box.text_frame.paragraphs[0]
    p.text = "NATIONAL INSTITUTE OF TECHNOLOGY (NIT) RAIPUR  •  DEPARTMENT OF INFORMATION TECHNOLOGY"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT

    # Main Title
    title_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.2), Inches(1.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Attention U-Net for Glacier Segmentation"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE

    p2 = tf.add_paragraph()
    p2.text = "Multispectral Deep Learning & Cross-Sensor Statistical Benchmarking in the Himalayas"
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLOR_CYAN_ACCENT
    p2.space_before = Pt(8)

    # Project Details Card
    card1 = add_card(s1, Inches(1.2), Inches(3.6), Inches(6.8), Inches(2.5))
    tf_card = card1.text_frame
    tf_card.word_wrap = True
    p = tf_card.paragraphs[0]
    p.text = "ACADEMIC PROJECT INFORMATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_card, "Degree & Semester:", "B.Tech Information Technology, 7th Semester", 11.5, 4)
    add_bullet_point(tf_card, "Project Category:", "B.Tech Minor Project (Phase-II Implementation & Audit)", 11.5, 4)
    add_bullet_point(tf_card, "Project Members:", "Rameshwar Chandravanshi  •  Deepanshu Sahu  •  Piyush Verma", 11.5, 4)
    add_bullet_point(tf_card, "Target Study Basin:", "Gangotri Glacier Catchment (Garhwal Himalaya, RGI Region 15)", 11.5, 4)

    # Right Image Card: False Color Composite
    img_card = add_card(s1, Inches(8.3), Inches(1.6), Inches(4.3), Inches(4.5))
    preview_img = os.path.join(ASSETS_DIR, "prediction_preview.png")
    if os.path.exists(preview_img):
        s1.shapes.add_picture(preview_img, Inches(8.4), Inches(1.7), Inches(4.1), Inches(4.3))

    add_slide_footer(s1, 1)

    # =========================================================================
    # SLIDE 2: Abstract
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, prs)
    add_slide_header(s2, "Section A: Introduction", "Project Abstract: Deep Learning Cryospheric Mapping",
                     "Executive overview of the problem, multispectral pipeline, and empirical outcomes")

    # 3 Summary Cards
    cards_data = [
        ("The Glaciological Challenge", [
            ("Alpine Water Security:", "Himalayan glaciers supply perennial meltwater to over 800 million people across South Asia."),
            ("Operational Deficiencies:", "Manual glacier delineations are labor-intensive and subjective; classical thresholding fails over debris-covered ice."),
            ("Topographic Noise:", "Steep north-facing cast shadows and adjacent granitic moraines confound single-band optical methods.")
        ], COLOR_ROSE_ACCENT),
        ("Proposed Deep Learning Solution", [
            ("8-Channel Input Tensor:", "Integrates 6 optical surface reflectance bands plus NDSI and NDVI physical indices."),
            ("CBAM Attention U-Net:", "Integrates Attention Gates with Channel and Spatial Attention (CBAM) to suppress background rock."),
            ("Focal Tversky Loss:", "Heavily penalizes false negatives along debris-covered margins (alpha=0.3, beta=0.7).")
        ], COLOR_CYAN_ACCENT),
        ("Verified Empirical Outcomes", [
            ("Model Segmentation Score:", "Achieved 0.7966 Overall Dice and 0.6619 IoU on genuine 30m Gangotri cryospheric patches."),
            ("Cross-Sensor Evaluation:", "Sentinel-2 MSI (Dice: 0.7838) outperforms Landsat 9 OLI (Dice: 0.6236) by +0.1601 on common 30m grid."),
            ("Statistical Rigor:", "Wilcoxon Signed-Rank Test proves statistically significant sensor degradation (p = 3.58e-07, Cohen's d = 2.115).")
        ], COLOR_GREEN_ACCENT)
    ]

    for i, (ctitle, cbullets, accent_col) in enumerate(cards_data):
        cx = Inches(0.8 + i * 3.95)
        add_card(s2, cx, Inches(1.8), Inches(3.8), Inches(4.9), border_color=accent_col)
        cbox = s2.shapes.add_textbox(cx + Inches(0.15), Inches(1.95), Inches(3.5), Inches(4.6))
        tf = cbox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ctitle
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = accent_col
        p.space_after = Pt(12)

        for bpre, btext in cbullets:
            add_bullet_point(tf, bpre, btext, pt_size=10.5, space_after=12, prefix_color=accent_col)

    add_slide_footer(s2, 2)

    # =========================================================================
    # SLIDE 3: Problem Statement
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, prs)
    add_slide_header(s3, "Section A: Introduction", "Problem Statement & Cryospheric Challenges",
                     "Why conventional remote sensing algorithms fail in high-relief alpine environments")

    add_card(s3, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))
    pbox = s3.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = pbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CORE REMOTE SENSING BOTTLENECK"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(10)

    add_bullet_point(tf, "1. Supraglacial Debris Cover:",
                     "Ablation zones in Himalayan valley glaciers are mantled by thick rocky moraines. Debris-covered ice exhibits spectral reflectance virtually identical to surrounding bedrock.", 11, 10)
    add_bullet_point(tf, "2. Extreme Topographic Shadows:",
                     "Steep Himalayan valley walls cast severe, multi-kilometer terrain shadows. Shadowed snow and ice reflect minimal solar irradiance, causing thresholding methods to fail.", 11, 10)
    add_bullet_point(tf, "3. Glacier Lake & Water Confusion:",
                     "Proglacial and supraglacial melt ponds share similar optical absorption characteristics with ice, leading to severe false positives.", 11, 10)
    add_bullet_point(tf, "4. Climate-Induced Hazards:",
                     "Rapid cryospheric retreat exacerbates Glacial Lake Outburst Flood (GLOF) risks, demanding automated, high-frequency boundary monitoring.", 11, 10)

    # Right side: Visual spatial discrepancy map
    add_card(s3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    disc_img = os.path.join(ASSETS_DIR, "spatial_discrepancy_map.png")
    if os.path.exists(disc_img):
        s3.shapes.add_picture(disc_img, Inches(6.9), Inches(1.9), Inches(5.5), Inches(4.5))

    add_slide_footer(s3, 3)

    # =========================================================================
    # SLIDE 4: Project Objectives
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, prs)
    add_slide_header(s4, "Section A: Introduction", "Research Objectives & Technical Scope",
                     "Target milestones designed and executed across the engineering and remote sensing framework")

    objs = [
        ("Objective 1: Multispectral Feature Fusion",
         "Construct an 8-channel input pipeline integrating optical visible/NIR/SWIR reflectance with physics-based indices (NDSI and NDVI) to enhance boundary separability.",
         COLOR_CYAN_ACCENT),
        ("Objective 2: Attention-Gated Neural Architecture",
         "Engineer an Attention U-Net incorporating Convolutional Block Attention Modules (CBAM) into skip connections to suppress non-glacier valley noise.",
         COLOR_GREEN_ACCENT),
        ("Objective 3: Class-Imbalance Loss Optimization",
         "Implement Focal Tversky Loss and Hybrid BCE-Dice formulations to penalize false negatives and eliminate omission of debris-covered glacier boundaries.",
         COLOR_GOLD_ACCENT),
        ("Objective 4: Cross-Sensor Statistical Benchmarking",
         "Execute a formal statistical comparison between Sentinel-2 MSI (S30) and Landsat 9 OLI (L30) on a common 30m grid using bootstrap CIs and Bland-Altman agreement.",
         COLOR_ROSE_ACCENT)
    ]

    for i, (otitle, odesc, ocol) in enumerate(objs):
        oy = Inches(1.8 + i * 1.25)
        add_card(s4, Inches(0.8), oy, Inches(11.7), Inches(1.1), border_color=ocol)
        obox = s4.shapes.add_textbox(Inches(1.0), oy + Inches(0.1), Inches(11.3), Inches(0.9))
        tf = obox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = otitle
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ocol
        p.space_after = Pt(4)

        p2 = tf.add_paragraph()
        p2.text = odesc
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_BODY

    add_slide_footer(s4, 4)

    # =========================================================================
    # SLIDE 5: Study Area & Reference Ground Truth
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, prs)
    add_slide_header(s5, "Section B: Data & Preprocessing", "Study Area: Gangotri Basin & RGI Region 15",
                     "Geographical basin context, glaciological specifications, and reference inventory validation")

    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))
    sbox = s5.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = sbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "GEOGRAPHIC & GLACIOLOGICAL SETTING"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "Study Catchment:", "Gangotri Glacier Basin, Uttarkashi, Uttarakhand, India (Garhwal Himalaya).", 10.5, 6)
    add_bullet_point(tf, "Geographic Bounding Box:", "[79.05°E - 79.30°E, 30.80°N - 31.05°N]. Extent: 23.5 km x 27.5 km.", 10.5, 6)
    add_bullet_point(tf, "Elevation Range:", "4,000 m (Gaumukh snout) up to 7,138 m (Chaukhamba I accumulation peak).", 10.5, 6)
    add_bullet_point(tf, "Hydrological Significance:", "Primary source of Bhagirathi River; one of the largest Himalayan valley glaciers (~30 km length, >120 km² area).", 10.5, 6)
    add_bullet_point(tf, "RGI Region 15 Resolution:",
                     "Gangotri is officially cataloged in Randolph Glacier Inventory (RGI 6.0) Region 15: South Asia East (Glacier ID: RGI60-15.06881). Region 14 terminates westward in Himachal Pradesh.", 10.5, 6)
    add_bullet_point(tf, "Basin Inventory Outlines:", "100 intersecting glacier outlines extracted from OGGM/GLIMS mirror and rasterized.", 10.5, 6)

    # Right side: Study area map
    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    map_img = os.path.join(ASSETS_DIR, "study_area_map.png")
    if os.path.exists(map_img):
        s5.shapes.add_picture(map_img, Inches(6.9), Inches(1.85), Inches(5.5), Inches(4.75))

    add_slide_footer(s5, 5)

    # =========================================================================
    # SLIDE 6: Sentinel-2 Multispectral Bands
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, prs)
    add_slide_header(s6, "Section B: Data & Preprocessing", "The 8-Channel Multispectral Input Tensor",
                     "Physical rationale for spectral band selection and spectral index inclusion")

    add_card(s6, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.9))
    bands_img = os.path.join(ASSETS_DIR, "multispectral_bands_diagram.png")
    if os.path.exists(bands_img):
        s6.shapes.add_picture(bands_img, Inches(0.9), Inches(1.85), Inches(11.5), Inches(4.75))

    add_slide_footer(s6, 6)

    # =========================================================================
    # SLIDE 7: Spectral Indices
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, prs)
    add_slide_header(s7, "Section B: Data & Preprocessing", "Spectral Indices: NDSI & NDVI Physical Foundations",
                     "Harnessing physical absorption signatures to separate snow, ice, bedrock, and vegetation")

    # Left: NDSI Card
    add_card(s7, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_CYAN_ACCENT)
    nbox = s7.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_n = nbox.text_frame
    tf_n.word_wrap = True
    p = tf_n.paragraphs[0]
    p.text = "1. NORMALIZED DIFFERENCE SNOW INDEX (NDSI)"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_n, "Mathematical Formula:", "NDSI = (Green - SWIR1) / (Green + SWIR1)\nSentinel-2: (B3 - B11) / (B3 + B11)\nLandsat 9: (B3 - B6) / (B3 + B6)", 11, 8)
    add_bullet_point(tf_n, "Physical Principle:", "Snow and clean ice exhibit high reflectance (>0.85) in visible wavelengths (Green) but absorb nearly all radiation in SWIR1 (~1.6 um).", 11, 8)
    add_bullet_point(tf_n, "Diagnostic Thresholding:", "Water clouds maintain high reflectance in both Green and SWIR1, allowing unambiguous cloud-ice discrimination.", 11, 8)
    add_bullet_point(tf_n, "Tensor Representation:", "Rescaled from [-1, 1] to [0, 1] as Channel 7.", 11, 8)

    # Right: NDVI Card
    add_card(s7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_GREEN_ACCENT)
    vbox = s7.shapes.add_textbox(Inches(7.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_v = vbox.text_frame
    tf_v.word_wrap = True
    p = tf_v.paragraphs[0]
    p.text = "2. NORMALIZED DIFFERENCE VEGETATION INDEX (NDVI)"
    p.font.size = Pt(12.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_v, "Mathematical Formula:", "NDVI = (NIR - Red) / (NIR + Red)\nSentinel-2: (B8 - B4) / (B8 + B4)\nLandsat 9: (B5 - B4) / (B5 + B4)", 11, 8)
    add_bullet_point(tf_v, "Physical Principle:", "Chlorophyll strongly absorbs Red light (~665 nm) and reflects Near-Infrared (~842 nm) due to mesophyll cell scattering.", 11, 8)
    add_bullet_point(tf_v, "Glaciological Role:", "Identifies sub-alpine valley vegetation along lateral moraines, eliminating false boundary expansions into scrubland.", 11, 8)
    add_bullet_point(tf_v, "Tensor Representation:", "Rescaled from [-1, 1] to [0, 1] as Channel 8.", 11, 8)

    add_slide_footer(s7, 7)

    # =========================================================================
    # SLIDE 8: Data Preprocessing Workflow
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, prs)
    add_slide_header(s8, "Section B: Data & Preprocessing", "Automated STAC Acquisition & Harmonization Pipeline",
                     "End-to-end data pipeline from public cloud COG archives to co-registered 30m tensor patches")

    add_card(s8, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.9))
    pipe_img = os.path.join(ASSETS_DIR, "preprocessing_pipeline_flowchart.png")
    if os.path.exists(pipe_img):
        s8.shapes.add_picture(pipe_img, Inches(0.9), Inches(1.85), Inches(11.5), Inches(4.75))

    add_slide_footer(s8, 8)

    # =========================================================================
    # SLIDE 9: Classical U-Net Foundations
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, prs)
    add_slide_header(s9, "Section C: Model Architecture", "U-Net Architectural Foundations for Segmentation",
                     "Encoder-decoder symmetry, multi-scale feature downsampling, and skip-connection topology")

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))
    ubox = s9.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = ubox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "CORE U-NET STRUCTURAL DESIGN"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "1. Contracting Encoder Path:",
                     "Successive 3x3 double convolutions followed by 2x2 max-pooling (stride 2). Progressively increases channel depth (64 -> 128 -> 256 -> 512) while reducing spatial resolution.", 11, 8)
    add_bullet_point(tf, "2. High-Level Bottleneck:",
                     "DoubleConv(512, 1024) at 16x16 spatial resolution, capturing abstract global contextual relationships across the entire glacier basin.", 11, 8)
    add_bullet_point(tf, "3. Expanding Decoder Path:",
                     "Transposed convolutions (ConvTranspose2d) progressively upsample feature maps back to original 256x256 dimensions.", 11, 8)
    add_bullet_point(tf, "4. Skip Connections:",
                     "Bridge encoder features directly to decoder stages, preserving localized spatial boundary details lost during pooling operations.", 11, 8)

    # Right card: Architectural strengths & limitations
    add_card(s9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    rbox = s9.shapes.add_textbox(Inches(6.95), Inches(1.95), Inches(5.4), Inches(4.6))
    tf_r = rbox.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "THE SKIP CONNECTION DILEMMA IN CRYOSPHERIC MAPPING"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_GOLD_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_r, "The Classical Dilemma:",
                     "Standard U-Net simply copies raw encoder features into the decoder via concatenation. In rugged alpine terrain, this transfers immense non-glacier background noise (moraines, shadows).", 11, 10)
    add_bullet_point(tf_r, "Redundant Features:",
                     "Low-level feature maps contain high activations for granitic valley ridges and clouds, confusing the decoder.", 11, 10)
    add_bullet_point(tf_r, "The Necessary Upgrade:",
                     "Attention Gates (AGs) and CBAM modules are integrated directly into skip connections to selectively filter activations before concatenation.", 11, 10)

    add_slide_footer(s9, 9)

    # =========================================================================
    # SLIDE 10: Attention Gates
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, prs)
    add_slide_header(s10, "Section C: Model Architecture", "Attention Gates (AG): Selective Spatial Filtering",
                     "Suppressing background noise and highlighting salient cryospheric boundaries")

    add_card(s10, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))
    agbox = s10.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = agbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "ATTENTION GATE MATHEMATICAL FORMULATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "Input Components:",
                     "Encoder feature map x (spatial features) and decoder gating signal g (rich contextual guidance from coarser level).", 11, 8)
    add_bullet_point(tf, "Intermediate Alignment:",
                     "Both inputs pass through 1x1 convolutions: W_x(x) and W_g(g), matching channel dimensions to intermediate depth F_int.", 11, 8)
    add_bullet_point(tf, "Activation & Attention Coefficient:",
                     "psi = Sigmoid( psi_conv( ReLU( W_x(x) + W_g(g) ) ) )\nProduces spatial attention coefficients psi in [0, 1].", 11, 8)
    add_bullet_point(tf, "Attention-Weighted Gating:",
                     "x_hat = x * psi. Pixels representing non-glacier terrain are suppressed towards 0, while cryospheric boundaries are scaled towards 1.", 11, 8)

    # Right side: AG & CBAM visual diagram
    add_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    ag_img = os.path.join(ASSETS_DIR, "attention_gate_cbam_diagram.png")
    if os.path.exists(ag_img):
        s10.shapes.add_picture(ag_img, Inches(6.9), Inches(1.85), Inches(5.5), Inches(4.75))

    add_slide_footer(s10, 10)

    # =========================================================================
    # SLIDE 11: CBAM Attention Mechanism
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, prs)
    add_slide_header(s11, "Section C: Model Architecture", "CBAM: Channel & Spatial Attention Synergy",
                     "Dual-attention sub-modules dynamically recalibrating inter-channel and intra-spatial features")

    # Left: Channel Attention
    add_card(s11, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_CYAN_ACCENT)
    cbox = s11.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_c = cbox.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "1. CHANNEL ATTENTION MODULE (CAM)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_c, "Operational Question:", "'WHAT spectral features are most meaningful?'", 11, 8)
    add_bullet_point(tf_c, "Dual Pooling:", "Applies both AdaptiveAvgPool2d(1) and AdaptiveMaxPool2d(1) to aggregate spatial context.", 11, 8)
    add_bullet_point(tf_c, "Shared Multi-Layer Perceptron:", "Routes pooled vectors through a shared bottleneck MLP (reduction ratio r=16) with ReLU activation.", 11, 8)
    add_bullet_point(tf_c, "Formulation:", "M_c(F) = Sigmoid( MLP(AvgPool(F)) + MLP(MaxPool(F)) )", 11, 8)
    add_bullet_point(tf_c, "Glaciological Effect:", "Prioritizes SWIR absorption and NDSI contrast channels over ambiguous visible bands.", 11, 8)

    # Right: Spatial Attention
    add_card(s11, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_GOLD_ACCENT)
    sbox = s11.shapes.add_textbox(Inches(7.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_s = sbox.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "2. SPATIAL ATTENTION MODULE (SAM)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_GOLD_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_s, "Operational Question:", "'WHERE are the salient glacier regions located?'", 11, 8)
    add_bullet_point(tf_s, "Channel Aggregation:", "Computes average-pooling and max-pooling across the channel axis: concat[AvgPool(F), MaxPool(F)] (2 channels).", 11, 8)
    add_bullet_point(tf_s, "Large Receptive Field:", "Applies a 7x7 convolution with padding 3 to capture broad spatial spatial neighborhood relationships.", 11, 8)
    add_bullet_point(tf_s, "Formulation:", "M_s(F') = Sigmoid( Conv7x7( [AvgPool(F'); MaxPool(F')] ) )", 11, 8)
    add_bullet_point(tf_s, "Glaciological Effect:", "Focuses activations along delicate glacier snouts and crevasses while ignoring barren valley floors.", 11, 8)

    add_slide_footer(s11, 11)

    # =========================================================================
    # SLIDE 12: Complete Model Architecture
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, prs)
    add_slide_header(s12, "Section C: Model Architecture", "Complete 8-Channel Attention U-Net Architecture",
                     "End-to-end network data flow matching models/attention_unet.py implementation")

    add_card(s12, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.9))
    arch_img = os.path.join(ASSETS_DIR, "complete_architecture_diagram.png")
    if os.path.exists(arch_img):
        s12.shapes.add_picture(arch_img, Inches(0.9), Inches(1.85), Inches(11.5), Inches(4.75))

    add_slide_footer(s12, 12)

    # =========================================================================
    # SLIDE 13: Training Methodology
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, prs)
    add_slide_header(s13, "Section D: Training & Evaluation", "Model Training Methodology & Hyperparameters",
                     "Hardware execution, optimization framework, and convergence controls")

    # Left: Training parameters table/card
    add_card(s13, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))
    tbox = s13.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TRAINING PARAMETER SPECIFICATIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "Dataset Size & Dimensions:", "24 genuine 30m cryospheric patches (256x256x8) and binary masks (256x256).", 10.5, 6)
    add_bullet_point(tf, "Train / Validation Split:", "80% Training (20 patches) | 20% Validation (4 patches).", 10.5, 6)
    add_bullet_point(tf, "Hardware Acceleration:", "Apple Silicon Metal Performance Shaders (MPS GPU).", 10.5, 6)
    add_bullet_point(tf, "Optimizer:", "AdamW (lr = 0.0005, beta1 = 0.9, beta2 = 0.999, weight_decay = 0.0001).", 10.5, 6)
    add_bullet_point(tf, "Batch Size & Epochs:", "Batch Size = 4 | Total Training Epochs = 15.", 10.5, 6)
    add_bullet_point(tf, "Loss Objective:", "Hybrid BCE-Dice Loss (alpha = 0.5) & Focal Tversky Loss.", 10.5, 6)
    add_bullet_point(tf, "Checkpointing:", "Validation Dice tracked each epoch; saved best checkpoint at Epoch 14 (Dice: 0.7574).", 10.5, 6)

    # Right: Reproducibility & Seeding Card
    add_card(s13, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    rbox = s13.shapes.add_textbox(Inches(6.95), Inches(1.95), Inches(5.4), Inches(4.6))
    tf_r = rbox.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "SCIENTIFIC RIGOR & MANDATORY CHECKS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_r, "Mandatory Weight Verification:",
                     "Evaluation and inference scripts strictly require a valid trained checkpoint file ('best_model.pth'). Evaluation on random uninitialized weights is prohibited and halts execution.", 11, 10)
    add_bullet_point(tf_r, "Strict Reproducibility:",
                     "Random seeds initialized across PyTorch, NumPy, and Python standard libraries (seed = 42).", 11, 10)
    add_bullet_point(tf_r, "Non-Overlapping Split:",
                     "Validation patches strictly isolated geographically to prevent spatial data leakage.", 11, 10)
    add_bullet_point(tf_r, "History Serialization:",
                     "Every training step logged to CSV and plotted into convergence curves automatically.", 11, 10)

    add_slide_footer(s13, 13)

    # =========================================================================
    # SLIDE 14: Loss Formulations
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, prs)
    add_slide_header(s14, "Section D: Training & Evaluation", "Loss Formulations: Addressing Class Imbalance",
                     "Mathematical formulation of Hybrid BCE-Dice and Focal Tversky Loss for boundary recovery")

    # Left: Hybrid BCE-Dice
    add_card(s14, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_CYAN_ACCENT)
    lbox = s14.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_l = lbox.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "1. HYBRID BCE-DICE LOSS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_l, "Formulation:", "L_Hybrid = alpha * L_BCE(p, y) + (1 - alpha) * L_Dice(p, y)", 11, 8)
    add_bullet_point(tf_l, "Pixel-Level Stability (BCE):", "L_BCE provides smooth pixel-wise gradients across the sigmoid activation surface.", 11, 8)
    add_bullet_point(tf_l, "Global Overlap (Dice):", "L_Dice = 1 - (2 * sum(p * y) + eps) / (sum(p) + sum(y) + eps). Directly optimizes area overlap regardless of foreground percentage.", 11, 8)
    add_bullet_point(tf_l, "Hyperparameter:", "alpha = 0.5 balances smooth local gradients with global boundary coherence.", 11, 8)

    # Right: Focal Tversky
    add_card(s14, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_ROSE_ACCENT)
    rbox = s14.shapes.add_textbox(Inches(7.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_r = rbox.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "2. FOCAL TVERSKY LOSS (DEBRIS RECOVERY)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ROSE_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_r, "Tversky Index (TI):", "TI = (TP + eps) / (TP + alpha * FP + beta * FN + eps)", 11, 8)
    add_bullet_point(tf_r, "Focal Formulation:", "L_FTL = (1 - TI)^gamma", 11, 8)
    add_bullet_point(tf_r, "Penalizing False Negatives:", "Setting beta = 0.7 and alpha = 0.3 forces the optimizer to heavily penalize missed glacier pixels (False Negatives).", 11, 8)
    add_bullet_point(tf_r, "Hard Example Mining (gamma = 1.33):", "Focuses gradient backpropagation on elusive, debris-mantled ablation margins rather than easy accumulation snow.", 11, 8)

    add_slide_footer(s14, 14)

    # =========================================================================
    # SLIDE 15: Evaluation Metrics
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, prs)
    add_slide_header(s15, "Section D: Training & Evaluation", "Quantitative Segmentation Evaluation Metrics",
                     "Standardized glaciological and computer vision performance criteria")

    metrics_cards = [
        ("Dice Similarity Coefficient (F1 Score)",
         "Dice = 2*TP / (2*TP + FP + FN)",
         "Measures spatial harmonic overlap between predicted glacier mask and RGI reference ground truth. Penalizes both over-segmentation and under-segmentation.",
         COLOR_CYAN_ACCENT),
        ("Intersection over Union (Jaccard Index)",
         "IoU = TP / (TP + FP + FN) = Dice / (2 - Dice)",
         "Strict spatial area overlap metric. Widely regarded as the benchmark standard in remote sensing land-cover segmentation benchmarks.",
         COLOR_GREEN_ACCENT),
        ("Precision (Positive Predictive Value)",
         "Precision = TP / (TP + FP)",
         "Quantifies over-segmentation. High precision guarantees that non-glacier valley bedrock and surrounding scrubland are not falsely classified as ice.",
         COLOR_GOLD_ACCENT),
        ("Recall (Sensitivity / True Positive Rate)",
         "Recall = TP / (TP + FN)",
         "Quantifies cryospheric completeness. High recall ensures that debris-covered ablation ice and shaded margins are not omitted.",
         COLOR_ROSE_ACCENT)
    ]

    for i, (mtitle, mform, mdesc, mcol) in enumerate(metrics_cards):
        cx = Inches(0.8 + (i % 2) * 5.95)
        cy = Inches(1.8 + (i // 2) * 2.5)
        add_card(s15, cx, cy, Inches(5.75), Inches(2.35), border_color=mcol)
        mbox = s15.shapes.add_textbox(cx + Inches(0.15), cy + Inches(0.1), Inches(5.45), Inches(2.15))
        tf = mbox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = mtitle
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = mcol

        p_form = tf.add_paragraph()
        p_form.text = mform
        p_form.font.size = Pt(11)
        p_form.font.bold = True
        p_form.font.color.rgb = COLOR_TEXT_WHITE
        p_form.space_before = Pt(3)

        p_desc = tf.add_paragraph()
        p_desc.text = mdesc
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = COLOR_TEXT_BODY
        p_desc.space_before = Pt(4)

    add_slide_footer(s15, 15)

    # =========================================================================
    # SLIDE 16: Training Results
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16, prs)
    add_slide_header(s16, "Section D: Training & Evaluation", "Training Convergence & Metric Progression",
                     "Empirical loss reduction and validation Dice improvement across 15 training epochs")

    # Left: Training curves plot
    add_card(s16, Inches(0.8), Inches(1.8), Inches(7.5), Inches(4.9))
    curves_img = os.path.join(ASSETS_DIR, "training_curves.png")
    if os.path.exists(curves_img):
        s16.shapes.add_picture(curves_img, Inches(0.9), Inches(1.85), Inches(7.3), Inches(4.75))

    # Right: Convergence summary card
    add_card(s16, Inches(8.5), Inches(1.8), Inches(4.0), Inches(4.9))
    tbox = s16.shapes.add_textbox(Inches(8.65), Inches(1.95), Inches(3.7), Inches(4.6))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TRAINING CONVERGENCE INSIGHTS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "Epoch 1-5 (Exploration):", "Model rapidly ascends from Dice 0.0000 to 0.4019 as convolutional kernels adapt to cryospheric spatial textures.", 10, 6)
    add_bullet_point(tf, "Epoch 6-10 (Stabilization):", "Validation Dice surpasses 0.6877; attention gates suppress lateral moraine background activations.", 10, 6)
    add_bullet_point(tf, "Epoch 14 (Global Peak):", "Best checkpoint achieved:\n• Val Dice: 0.7574\n• Val IoU: 0.6095\n• Val Precision: 0.8349\n• Val Recall: 0.6930\n• Val Loss: 0.3037", 10.5, 6, prefix_color=COLOR_GREEN_ACCENT)
    add_bullet_point(tf, "Overfitting Resistance:", "Smooth validation curves confirm robust generalization without divergence or catastrophic memorization.", 10, 6)

    add_slide_footer(s16, 16)

    # =========================================================================
    # SLIDE 17: Segmentation Results
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    set_slide_background(s17, prs)
    add_slide_header(s17, "Section D: Training & Evaluation", "Glacier Segmentation Evaluation on Genuine Patches",
                     "Overall quantitative benchmarks and multi-panel spatial error discrepancy breakdown")

    # Left: Overlay preview figure
    add_card(s17, Inches(0.8), Inches(1.8), Inches(7.5), Inches(4.9))
    overlay_img = os.path.join(ASSETS_DIR, "segmentation_overlay_preview.png")
    if os.path.exists(overlay_img):
        s17.shapes.add_picture(overlay_img, Inches(0.9), Inches(1.85), Inches(7.3), Inches(4.75))

    # Right: Results Table & Confusion Matrix
    add_card(s17, Inches(8.5), Inches(1.8), Inches(4.0), Inches(4.9))
    rbox = s17.shapes.add_textbox(Inches(8.65), Inches(1.95), Inches(3.7), Inches(4.6))
    tf = rbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "BENCHMARK PERFORMANCE (N=24)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "Overall Dice:", "0.7966 (Mean per-patch: 0.7838)", 10.5, 5)
    add_bullet_point(tf, "Overall IoU:", "0.6619 (Mean per-patch: 0.6494)", 10.5, 5)
    add_bullet_point(tf, "Overall Precision:", "0.8810 (Minimal over-segmentation)", 10.5, 5)
    add_bullet_point(tf, "Overall Recall:", "0.7269 (Strong boundary capture)", 10.5, 5)
    add_bullet_point(tf, "Overall F1 Score:", "0.7966", 10.5, 5)

    p_div = tf.add_paragraph()
    p_div.text = "CONFUSION MATRIX (1,572,864 PIXELS)"
    p_div.font.size = Pt(11)
    p_div.font.bold = True
    p_div.font.color.rgb = COLOR_GOLD_ACCENT
    p_div.space_before = Pt(6)
    p_div.space_after = Pt(4)

    add_bullet_point(tf, "True Positives (TP):", "338,340 pixels (Agreed ice)", 10, 3)
    add_bullet_point(tf, "False Positives (FP):", "45,688 pixels (Over-segmented)", 10, 3)
    add_bullet_point(tf, "False Negatives (FN):", "127,142 pixels (Omitted ice)", 10, 3)
    add_bullet_point(tf, "True Negatives (TN):", "1,061,694 pixels (Background)", 10, 3)

    add_slide_footer(s17, 17)

    # =========================================================================
    # SLIDE 18: Multi-Satellite Comparison Framework
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    set_slide_background(s18, prs)
    add_slide_header(s18, "Section E: Statistical Analysis", "Cross-Sensor Comparison: Sentinel-2 vs. Landsat 9",
                     "Evaluating orbital transferability under the Harmonized Landsat-Sentinel (HLS S30 vs L30) protocol")

    # Sensor A Card
    add_card(s18, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_CYAN_ACCENT)
    sbox = s18.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_s = sbox.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "SENSOR A: SENTINEL-2B MSI (HLS S30)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_s, "Operating Agency:", "European Space Agency (ESA) Copernicus Programme.", 11, 7)
    add_bullet_point(tf_s, "Acquisition Scene ID:", "S2B_44RLV_20241015_0_L2A", 11, 7)
    add_bullet_point(tf_s, "Acquisition Timestamp:", "October 15, 2024 at 05:40 UTC", 11, 7)
    add_bullet_point(tf_s, "Cloud Cover:", "0.02% (Virtually cloud-free post-monsoon scene)", 11, 7)
    add_bullet_point(tf_s, "Native Resolution:", "10m (B2, B3, B4, B8) and 20m (B11, B12)", 11, 7)
    add_bullet_point(tf_s, "Harmonization Grid:", "Resampled to 30.0m HLS standard via Bilinear interpolation.", 11, 7)

    # Sensor B Card
    add_card(s18, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_GOLD_ACCENT)
    lbox = s18.shapes.add_textbox(Inches(7.0), Inches(1.95), Inches(5.3), Inches(4.6))
    tf_l = lbox.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "SENSOR B: LANDSAT 9 OLI-2 (HLS L30)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_GOLD_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_l, "Operating Agency:", "NASA / USGS Earth Observation Program.", 11, 7)
    add_bullet_point(tf_l, "Acquisition Scene ID:", "LC09_L2SP_146039_20241026_02_T1", 11, 7)
    add_bullet_point(tf_l, "Acquisition Timestamp:", "October 26, 2024 at 05:18 UTC", 11, 7)
    add_bullet_point(tf_l, "Cloud Cover:", "1.96% (Clear ablation conditions)", 11, 7)
    add_bullet_point(tf_l, "Native Resolution:", "30.0m native optical resolution across all bands.", 11, 7)
    add_bullet_point(tf_l, "Temporal Baseline:", "11-day delta in late autumn; identical cryospheric state.", 11, 7)
    add_bullet_point(tf_l, "Zero Synthetic Fallback:", "Strictly real satellite acquisitions; zero simulation.", 11, 7)

    add_slide_footer(s18, 18)

    # =========================================================================
    # SLIDE 19: Statistical Analysis Workflow
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    set_slide_background(s19, prs)
    add_slide_header(s19, "Section E: Statistical Analysis", "Cross-Sensor Statistical Analysis Protocol",
                     "Structured 8-stage statistical pipeline for rigorous non-parametric hypothesis testing")

    add_card(s19, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.9))
    stat_flow = os.path.join(ASSETS_DIR, "statistical_workflow_flowchart.png")
    if os.path.exists(stat_flow):
        s19.shapes.add_picture(stat_flow, Inches(0.9), Inches(1.85), Inches(11.5), Inches(4.75))

    add_slide_footer(s19, 19)

    # =========================================================================
    # SLIDE 20: Statistical Tests
    # =========================================================================
    s20 = prs.slides.add_slide(blank_layout)
    set_slide_background(s20, prs)
    add_slide_header(s20, "Section E: Statistical Analysis", "Statistical Hypothesis Tests & Agreement Metrics",
                     "Mathematical formulation and interpretation of applied inferential statistics")

    tests_data = [
        ("Shapiro-Wilk Normality Test",
         "W = 0.949, p = 0.2522",
         "Evaluates whether paired differences between sensors follow a Gaussian distribution. Since p > 0.05, parametric tests are valid, but non-parametric tests are also applied for robust verification.",
         COLOR_CYAN_ACCENT),
        ("Wilcoxon Signed-Rank Test",
         "W = 2.0, p = 3.58e-07 (Significant)",
         "Non-parametric paired rank test. Proves that the positive difference favoring Sentinel-2 over Landsat 9 is statistically significant (p < 0.001) and not due to stochastic variation.",
         COLOR_GREEN_ACCENT),
        ("Effect Size: Cohen's d & Cliff's delta",
         "Cohen's d = 2.115 | Cliff's delta = 0.861",
         "Quantifies the magnitude of performance divergence. A Cohen's d of 2.115 represents an extremely large effect size, indicating substantial degradation upon cross-sensor transfer.",
         COLOR_GOLD_ACCENT),
        ("Bland-Altman Agreement Analysis",
         "Mean Bias = +0.1601 | 95% LoA: [+0.0118, +0.3085]",
         "Clinical/geodetic agreement metric. 95.8% of observations fall strictly within limits of agreement (+- 1.96 SD), proving consistent systematic bias across the entire catchment.",
         COLOR_ROSE_ACCENT)
    ]

    for i, (ttitle, tstat, tdesc, tcol) in enumerate(tests_data):
        cx = Inches(0.8 + (i % 2) * 5.95)
        cy = Inches(1.8 + (i // 2) * 2.5)
        add_card(s20, cx, cy, Inches(5.75), Inches(2.35), border_color=tcol)
        tbox = s20.shapes.add_textbox(cx + Inches(0.15), cy + Inches(0.1), Inches(5.45), Inches(2.15))
        tf = tbox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ttitle
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = tcol

        p_stat = tf.add_paragraph()
        p_stat.text = tstat
        p_stat.font.size = Pt(11)
        p_stat.font.bold = True
        p_stat.font.color.rgb = COLOR_TEXT_WHITE
        p_stat.space_before = Pt(3)

        p_desc = tf.add_paragraph()
        p_desc.text = tdesc
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = COLOR_TEXT_BODY
        p_desc.space_before = Pt(4)

    add_slide_footer(s20, 20)

    # =========================================================================
    # SLIDE 21: Statistical Graphs & Results
    # =========================================================================
    s21 = prs.slides.add_slide(blank_layout)
    set_slide_background(s21, prs)
    add_slide_header(s21, "Section E: Statistical Analysis", "Empirical Distribution & Uncertainty Visualizations",
                     "Boxplots, violin densities, 1:1 identity correlations, and 10,000-sample bootstrap forest plots")

    # 4 Graphs arranged in a 2x2 grid
    graph_grid = [
        (os.path.join(ASSETS_DIR, "dice_distribution.png"), Inches(0.8), Inches(1.8), Inches(5.75), Inches(2.35),
         "Dice Score Distributions (Boxplot & Split Violin)"),
        (os.path.join(ASSETS_DIR, "confidence_intervals.png"), Inches(6.8), Inches(1.8), Inches(5.75), Inches(2.35),
         "Forest Plot: 95% Bootstrap Confidence Intervals"),
        (os.path.join(ASSETS_DIR, "iou_comparison.png"), Inches(0.8), Inches(4.35), Inches(5.75), Inches(2.35),
         "IoU Sensor Correlation Scatter & Probability Density"),
        (os.path.join(ASSETS_DIR, "sensor_comparison_summary.png"), Inches(6.8), Inches(4.35), Inches(5.75), Inches(2.35),
         "Mean Metric Comparison Summary with 95% CI Error Bars")
    ]

    for gpath, gx, gy, gw, gh, gcaption in graph_grid:
        add_card(s21, gx, gy, gw, gh)
        if os.path.exists(gpath):
            s21.shapes.add_picture(gpath, gx + Inches(0.1), gy + Inches(0.1), gw - Inches(0.2), gh - Inches(0.2))

    add_slide_footer(s21, 21)

    # =========================================================================
    # SLIDE 22: Statistical Findings & Interpretation
    # =========================================================================
    s22 = prs.slides.add_slide(blank_layout)
    set_slide_background(s22, prs)
    add_slide_header(s22, "Section E: Statistical Analysis", "Statistical Findings & Physical Interpretation",
                     "Synthesizing quantitative results, agreement parameters, and sensor spectral response effects")

    # Left: Numerical findings card
    add_card(s22, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9))
    fbox = s22.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = fbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "SYNTHESIS OF EMPIRICAL FINDINGS (N=24)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "Sentinel-2 MSI (S30):", "Mean Dice = 0.7838 +- 0.070 | Median = 0.7949 (95% CI: [0.7557, 0.8101])", 10.5, 6)
    add_bullet_point(tf, "Landsat 9 OLI (L30):", "Mean Dice = 0.6236 +- 0.080 | Median = 0.6317 (95% CI: [0.5925, 0.6545])", 10.5, 6)
    add_bullet_point(tf, "Mean Paired Difference:", "+0.1601 Dice (+16.01% advantage for Sentinel-2, 95% CI: [+0.1292, +0.1882])", 10.5, 6, prefix_color=COLOR_GREEN_ACCENT)
    add_bullet_point(tf, "Mean IoU Difference:", "+0.1916 IoU (95% CI: [+0.1542, +0.2260])", 10.5, 6)
    add_bullet_point(tf, "Wilcoxon Signed-Rank Test:", "W = 2.0, p = 3.58e-07 (Statistically Significant at alpha=0.05)", 10.5, 6)
    add_bullet_point(tf, "Paired t-test:", "t = 10.363, p = 3.88e-10 (Statistically Significant)", 10.5, 6)
    add_bullet_point(tf, "Effect Size:", "Cohen's d = 2.115 (Huge Effect Size) | Cliff's delta = 0.861", 10.5, 6)
    add_bullet_point(tf, "Bland-Altman Agreement:", "Mean Bias = +0.1601; 95.8% (23/24) patches within Limits of Agreement.", 10.5, 6)

    # Right: Prediction comparison visual
    add_card(s22, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    pred_comp = os.path.join(ASSETS_DIR, "prediction_comparison.png")
    if os.path.exists(pred_comp):
        s22.shapes.add_picture(pred_comp, Inches(6.9), Inches(1.85), Inches(5.5), Inches(4.75))

    add_slide_footer(s22, 22)

    # =========================================================================
    # SLIDE 23: Limitations & Scientific Discussion
    # =========================================================================
    s23 = prs.slides.add_slide(blank_layout)
    set_slide_background(s23, prs)
    add_slide_header(s23, "Section F: Conclusion & Future Scope", "Glaciological Limitations & Scientific Discussion",
                     "Critical analysis of physical constraints and remote sensing failure modes")

    limits = [
        ("1. Supraglacial Debris Spectral Ambiguity",
         "Lower Gangotri ablation zone features extensive supraglacial rock debris derived from lateral rockfalls. Because supraglacial debris is composed of the same granitic petrology as adjacent valley walls, optical surface reflectance alone exhibits minimal spectral contrast.",
         COLOR_ROSE_ACCENT),
        ("2. Topographic Shadows in Steep Cirques",
         "Himalayan peaks (>6,500 m) cast profound shadows across north-facing accumulation zones. In deep shadow, signal-to-noise ratio decreases substantially, occasionally elevating false negative omission rates.",
         COLOR_GOLD_ACCENT),
        ("3. Cross-Sensor Relative Spectral Response (RSR) Shift",
         "Even when resampled to an identical 30.0m HLS grid, differences between Sentinel-2 MSI and Landsat 9 OLI Relative Spectral Response (RSR) filters (particularly in SWIR and NIR channels) induce domain shift when applying weights without fine-tuning.",
         COLOR_CYAN_ACCENT),
        ("4. Single-Basin Geographic Specialization",
         "The current model was trained on the Gangotri Glacier basin. Regional generalization across diverse cryospheric regimes (e.g., surge-type Karakoram glaciers or maritime Alps) requires cross-regional transfer evaluation.",
         COLOR_GREEN_ACCENT)
    ]

    for i, (ltitle, ldesc, lcol) in enumerate(limits):
        ly = Inches(1.8 + i * 1.25)
        add_card(s23, Inches(0.8), ly, Inches(11.7), Inches(1.15), border_color=lcol)
        lbox = s23.shapes.add_textbox(Inches(1.0), ly + Inches(0.08), Inches(11.3), Inches(0.95))
        tf = lbox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ltitle
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = lcol
        p.space_after = Pt(3)

        p2 = tf.add_paragraph()
        p2.text = ldesc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = COLOR_TEXT_BODY

    add_slide_footer(s23, 23)

    # =========================================================================
    # SLIDE 24: Conclusion & Future Scope
    # =========================================================================
    s24 = prs.slides.add_slide(blank_layout)
    set_slide_background(s24, prs)
    add_slide_header(s24, "Section F: Conclusion & Future Scope", "Project Summary, Future Scope & Acknowledgements",
                     "Core project contributions, forward-looking cryospheric research roadmap, and Q&A")

    # Left: Contributions Summary Card
    add_card(s24, Inches(0.8), Inches(1.8), Inches(5.8), Inches(4.9), border_color=COLOR_GREEN_ACCENT)
    cbox = s24.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(5.5), Inches(4.6))
    tf = cbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "SUMMARY OF KEY CONTRIBUTIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf, "1. Validated Deep Learning Pipeline:",
                     "Engineered an 8-channel Attention U-Net incorporating CBAM spatial/channel attention gates and Focal Tversky Loss for high-relief glacier segmentation.", 10.5, 6)
    add_bullet_point(tf, "2. Automated STAC Data Pipeline:",
                     "Implemented windowed Cloud-Optimized GeoTIFF streaming from AWS Earth Search and MS Planetary Computer, standardizing on a NASA HLS 30m grid.", 10.5, 6)
    add_bullet_point(tf, "3. Glaciological Integrity:",
                     "Corrected RGI cataloging to Region 15: South Asia East (Gangotri Glacier RGI60-15.06881) and eliminated all synthetic data fallbacks.", 10.5, 6)
    add_bullet_point(tf, "4. Rigorous Statistical Framework:",
                     "Proved statistically significant sensor transfer degradation (p = 3.58e-07, Cohen's d = 2.115) with 10,000 bootstrap CIs and Bland-Altman agreement.", 10.5, 6)

    # Right: Future Scope & Thank You Card
    add_card(s24, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9), border_color=COLOR_CYAN_ACCENT)
    fbox = s24.shapes.add_textbox(Inches(6.95), Inches(1.95), Inches(5.4), Inches(4.6))
    tf_f = fbox.text_frame
    tf_f.word_wrap = True
    p = tf_f.paragraphs[0]
    p.text = "FUTURE RESEARCH HORIZONS & ACKNOWLEDGEMENTS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN_ACCENT
    p.space_after = Pt(8)

    add_bullet_point(tf_f, "SAR Coherence Fusion:", "Integrate Sentinel-1 Synthetic Aperture Radar (C-band) coherence maps to penetrate supraglacial debris and clouds.", 10.5, 6)
    add_bullet_point(tf_f, "Thermal Infrared (TIRS) Integration:", "Utilize Landsat Thermal Infrared bands to estimate debris mantle thickness and sub-debris ablation rates.", 10.5, 6)
    add_bullet_point(tf_f, "Multi-Basin Transfer Learning:", "Evaluate regional generalization across Karakoram (Region 14), Western Himalaya, and European Alps.", 10.5, 6)
    add_bullet_point(tf_f, "Acknowledgements:", "Department of Information Technology, NIT Raipur; Project Supervisor; Open Science providers (ESA Copernicus, USGS/NASA, OGGM/GLIMS).", 10.5, 6)

    p_ty = tf_f.add_paragraph()
    p_ty.text = "THANK YOU  •  QUESTIONS & DISCUSSION"
    p_ty.font.size = Pt(13)
    p_ty.font.bold = True
    p_ty.font.color.rgb = COLOR_GOLD_ACCENT
    p_ty.space_before = Pt(12)
    p_ty.alignment = PP_ALIGN.CENTER

    p_repo = tf_f.add_paragraph()
    p_repo.text = "GitHub Repository: github.com/rameshwar17403/Attention-U-Net-Glacier-Segmentation"
    p_repo.font.size = Pt(9.5)
    p_repo.font.color.rgb = COLOR_TEXT_MUTED
    p_repo.alignment = PP_ALIGN.CENTER

    add_slide_footer(s24, 24)

    # Save to scratch and desktop
    ppt_path_scratch = os.path.join(OUTPUT_DIR_SCRATCH, "final_project_presentation.pptx")
    ppt_path_desktop = os.path.join(OUTPUT_DIR_DESKTOP, "final_project_presentation.pptx")

    prs.save(ppt_path_scratch)
    prs.save(ppt_path_desktop)
    print(f"[+] Successfully exported 24-slide PowerPoint presentation to:")
    print(f"    1. {ppt_path_desktop}")
    print(f"    2. {ppt_path_scratch}")


if __name__ == "__main__":
    build_presentation()
