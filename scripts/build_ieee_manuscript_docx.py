"""
Build Final IEEE-Style Word Manuscript (.DOCX), Figure Inventory, and Validation Report.
Strict adherence to:
- Source-of-truth project metrics
- IEEE conference two-column body layout
- 13 high-resolution publication figures embedded
- 8 comprehensive academic tables
- Natural student/researcher academic tone (no buzzwords, no overclaiming)
- Cautious, scientifically accurate reporting of non-significant results (p = 0.1927)
- Clear distinction between primary confirmatory (H=5) and exploratory (H=1) analyses
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table):
    """Apply clean academic IEEE table borders: top rule, header bottom rule, bottom rule."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>
            <w:insideV w:val="none"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def format_row(row, is_header=False, font_size=7.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT):
    for cell in row.cells:
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        if is_header:
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F2F2F2"/>')
            cell._tc.get_or_add_tcPr().append(shading)
        for p in cell.paragraphs:
            p.alignment = align
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(font_size)
                run.font.bold = bold or is_header

def add_ieee_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10.5)
    run.font.bold = True
    return p

def add_ieee_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.5)
    run.font.italic = True
    run.font.bold = True
    return p

def add_body_p(doc, text, space_after=3.5, indent=0.15):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.05
    p.paragraph_format.first_line_indent = Inches(indent)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.5)
    return p

def add_equation_p(doc, eq_text, eq_num_text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_eq = p.add_run(eq_text)
    r_eq.font.name = "Times New Roman"
    r_eq.font.size = Pt(9.5)
    r_eq.font.italic = True
    r_sp = p.add_run("   " * 3)
    r_num = p.add_run(eq_num_text)
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(9.5)
    return p

def add_figure_with_caption(doc, img_path, caption_num, caption_title, caption_desc, width_in=3.25):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.paragraph_format.keep_with_next = True
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(img_path, width=Inches(width_in))
    
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    r_num = p_cap.add_run(f"Fig. {caption_num}. ")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(8.5)
    r_num.font.bold = True
    
    r_title = p_cap.add_run(f"{caption_title}. ")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(8.5)
    r_title.font.italic = True
    
    r_desc = p_cap.add_run(caption_desc)
    r_desc.font.name = "Times New Roman"
    r_desc.font.size = Pt(8.0)

def add_table_title(doc, tbl_num, tbl_title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(f"TABLE {tbl_num}\n")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(8.5)
    r1.font.bold = True
    r2 = p.add_run(tbl_title.upper())
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(8.0)
    r2.font.bold = True

print("Script template ready.")
