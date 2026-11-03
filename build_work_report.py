"""수행 내역 보고서 생성 (표지·목차·선행연구 제외, 본 연구에서 수행한 작업만 기술)."""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = Path(__file__).resolve().parent
FIG = BASE / "보고서_그림"
OUT = BASE / "수행내역_보고서.docx"
FONT = "맑은 고딕"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)


def set_font(obj, size=None, bold=None):
    f = obj.font
    f.name = FONT
    el = obj.element if hasattr(obj, "element") else obj._element
    rpr = el.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), FONT)
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold


normal = doc.styles["Normal"]
set_font(normal, size=10.5)
normal.paragraph_format.line_spacing = 1.6
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for name, size in [("Heading 1", 13), ("Heading 2", 11.5)]:
    st = doc.styles[name]
    set_font(st, size=size, bold=True)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(5)
    st.paragraph_format.keep_with_next = True
    ppr = st.element.get_or_add_pPr()
    ol = ppr.find(qn("w:outlineLvl"))
    if ol is None:
        ol = OxmlElement("w:outlineLvl")
        ppr.append(ol)
    ol.set(qn("w:val"), "9")



from docx.enum.style import WD_STYLE_TYPE

SECTION_STYLES = {}
for _lvl, _size in [(1, 13), (2, 11.5)]:
    _st = doc.styles.add_style(f"절 제목 {_lvl}", WD_STYLE_TYPE.PARAGRAPH)
    _st.base_style = normal
    set_font(_st, size=_size, bold=True)
    _st.font.color.rgb = RGBColor(0, 0, 0)
    _st.paragraph_format.space_before = Pt(12)
    _st.paragraph_format.space_after = Pt(5)
    _st.paragraph_format.keep_with_next = True
    SECTION_STYLES[_lvl] = _st


def add_heading(text, level):
    p = doc.add_paragraph(style=SECTION_STYLES[level])
    p.add_run(text)
    return p


def para(text, bold=False, size=None, align=None, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    r.italic = italic
    if align is not None:
        p.alignment = align
    return p


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size=8.5)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


_tab = [0]
_fig = [0]


def caption_table(title):
    _tab[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"표 {_tab[0]}  ")
    set_font(r, size=9.5, bold=True)
    r2 = p.add_run(title)
    set_font(r2, size=9.5)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def _border(cell, edge, sz):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    borders.append(el)


def table(headers, rows, widths, aligns=None, size=8.5):
    """booktabs 스타일: 상·하단 굵은 선, 헤더 아래 얇은 선, 세로선 없음."""
    n = len(headers)
    aligns = aligns or ["L"] + ["C"] * (n - 1)
    t = doc.add_table(rows=1 + len(rows), cols=n)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    tb = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        tb.append(el)
    tblPr.append(tb)
    all_rows = [headers] + rows
    for i, row in enumerate(all_rows):
        for j, text in enumerate(row):
            c = t.rows[i].cells[j]
            c.width = Cm(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = {"L": WD_ALIGN_PARAGRAPH.LEFT, "C": WD_ALIGN_PARAGRAPH.CENTER,
                           "R": WD_ALIGN_PARAGRAPH.RIGHT}[aligns[j]]
            r = p.add_run(str(text))
            set_font(r, size=size, bold=(i == 0))
            if i == 0:
                _border(c, "top", 12)
                _border(c, "bottom", 6)
            if i == len(all_rows) - 1:
                _border(c, "bottom", 12)
    for j, w in enumerate(widths):
        t.columns[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(fname, width_cm, title):
    _fig[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / fname), width=Cm(width_cm))
    c = doc.add_paragraph()
    r = c.add_run(f"그림 {_fig[0]}  ")
    set_font(r, size=9.5, bold=True)
    r2 = c.add_run(title)
    set_font(r2, size=9.5)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)


# ---------- 제목 ----------
para("한국어 실리콘 샘플링의 타당성 검증을 위한 연구 수행 내역", bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER)

add_heading("1. 서론", 1)
para("본 장은 중간 보고 이후 본 연구에서 수행한 작업의 내용과 결과를 정리한다. 실리콘 샘플링에서 LLM의 응답 분포가 실제 분포와 달라지는 원인을 분해하기 위해, 자료 처리, 모델 추론 환경 검증, 분석, 오염 점검을 차례로 진행하였다. 각 절은 수행한 절차와 관측된 결과를 기술하며, 확정된 결론과 탐색적 관측을 구분하여 서술한다.")

add_heading("2. 자료 처리 및 문항 선정", 1)
para("분석 자료는 한국종합사회조사(KGSS) 2023년 회차이다. 결측 코드 -1은 비해당, -8은 무응답으로 처리하였다. -1은 해당 회차 미조사와 건너뛰기 규칙에 의한 비해당이 섞여 있으므로, 그 해에 -1이 아닌 값이 하나라도 있는 문항만 조사된 문항으로 판정하였다. 값 라벨의 8, 88, 9, 99는 통합 이전 회차의 잔여물로 보고, 무응답 판정을 코드값이 아니라 실질 선택지의 연속 구간으로 하였다.")
para("문항 후보는 응답률 90% 이상, 비해당 5% 이하, 선택지 2–7개의 기준으로 303개를 선정하였다. 이 중 본실험에 쓸 문항은 유형과 민감도를 기준으로 29개를 확정하였다. 세 명의 연구자가 문항 유형(사실, 가치관, 차별)과 민감 여부를 독립적으로 태깅하였고, 문항 유형은 다수결로 병합하였다.")
caption_table("문항 유형 태깅 결과 및 다수결 병합")
table(["문항 유형", "민감 Y", "민감 N", "공란", "합계"],
      [["가치관", "30", "164", "0", "194"], ["차별", "17", "0", "0", "17"],
       ["사실", "0", "0", "92", "92"], ["합계", "47", "164", "92", "303"]],
      [4.0, 3.0, 3.0, 3.0, 3.0])
note("주: 문항 유형의 Fleiss κ는 0.775, 민감 여부의 Fleiss κ는 0.424였다. 민감 여부의 일치도가 낮아 판정이 갈린 67문항은 재논의가 필요하다. 사실 문항 10개에 기록된 민감 N은 태깅 기준에 따라 공란으로 처리하였다.")
para("셀 최빈응답 정확도는 성별×연령 셀 최빈 기준으로 0.5385였다. 같은 기준의 전체 최빈 정확도는 0.5293이었다. 인구통계 순이익의 중앙값은 0으로, 303문항의 절반 이상에서 인구통계가 개인 예측을 개선하지 못하였다.")

add_heading("3. 모델 추론 환경 검증", 1)
para("EXAONE 3.5(2.4B)와 Kanana 1.5(2.1B)를 대상으로 선택지 토큰화와 로짓 접근을 검증하였다. 두 모델 모두에서 숫자 1–9가 단일 토큰으로 인코딩되었다. 앞 공백은 별도 토큰으로 분리되므로, 프롬프트를 \"정답: \"처럼 공백으로 끝내 prefill하는 방식을 채택하였다. 첫 토큰 확률질량은 EXAONE에서 0.983–1.0, Kanana에서 0.9998–1.0이었다.")
para("재현성을 확인하기 위해 Kanana의 C0 조건 255건을 런타임을 새로 받아 재실행하였다. 같은 GPU 종류와 라이브러리 버전에서 원소별 최대 절대차는 0이었다(2026년 10월 4일). 또한 EXAONE은 원격 코드 리비전(8e6fc27)을, 두 모델 모두 transformers 4.48.3을 고정하여 실행 환경을 통일하였다. 추가 후보인 Qwen2.5-3B-Instruct는 토큰화와 로짓 점검을 통과하였으나, gemma-4-e2b-it는 아키텍처 인식을 위해 별도 버전 고정이 필요하여 보류하였다.")

add_heading("4. 선택지 순서 효과의 인간 기준선 및 모델 배율", 1)
para("KGSS 2021·2023의 A/B 분할표본 16쌍 중 13쌍에서 선택지 배열이 역전되어 있었다. 인간 응답에서 배열 효과의 TV거리는 0.02–0.05였다. 국가자긍심 문항 10개 중 9개에서 효과 방향이 일치하였고, 우연으로 이 정도 일치할 확률은 약 1.1%였다. 방향은 뒤에 제시된 선택지가 더 선택되는 쪽이었다. 본 조사는 낭독과 보기카드를 혼합한 방식이므로, 이 방향의 이론적 예측은 자명하지 않으며 경험적 관측으로만 기술한다.")
para("같은 13쌍에 대해 모델의 TV거리를 인간 TV거리로 나눈 배율을 계산하였다(표 2, 그림 1). EXAONE의 배율 중앙값은 0.71이었고 4쌍에서 1을 넘었다. Kanana의 중앙값은 0.24였고 5쌍에서 1을 넘었다. 쌍별 표본이 13개에 불과하므로 방향성 관측으로만 해석한다.")
caption_table("선택지 순서 효과의 인간 기준선 대비 배율 (13쌍)")
table(["구분", "인간 TV거리(중앙값)", "모델 배율 중앙값", "배율 > 1인 쌍", "최대 배율"],
      [["EXAONE 3.5 2.4B", "0.036", "0.71", "4 / 13", "15.3 (PROUDPOL16)"],
       ["Kanana 1.5 2.1B", "0.036", "0.24", "5 / 13", "48.8 (PROUDPOL16)"]],
      [3.4, 3.2, 3.0, 2.8, 4.6], aligns=["L", "C", "C", "C", "L"])
figure("fig3_order_ratio.png", 13.5, "선택지 순서 효과의 모델 대 인간 배율 (로그 척도, 1은 인간과 같은 크기)")

add_heading("5. 배치 추론의 검증과 정밀도 점검", 1)
para("한 건씩 계산하던 로짓 추출을 배치 처리로 바꾸어 소요 시간을 줄이고자 하였다. 배치 처리에서는 왼쪽 패딩과 위치 번호를 맞추고, 마지막 위치의 로짓만 어휘 전체에 대해 계산하였다. 이 방식으로 배치 32에서 발생하던 메모리 부족 문제를 해소하였다. 판정 기준은 한 건씩 계산한 결과와의 최대 절대차 1e-3 미만, 최빈 선택 뒤집힘 0건으로 사전에 정하였다.")
caption_table("배치 추론 게이트 결과 (fp32, 프로브 200건)")
table(["설정", "EXAONE 시간(초)", "EXAONE 최대 차이", "Kanana 시간(초)", "Kanana 최대 차이", "최빈 뒤집힘", "판정"],
      [["한 건씩(기준)", "46.0", "–", "45.6", "–", "–", "–"],
       ["배치 8", "41.8", "1.60e-05", "40.5", "2.76e-05", "0 / 0", "통과"],
       ["배치 16", "46.1", "1.63e-05", "42.1", "3.38e-05", "0 / 0", "통과"],
       ["배치 32", "47.2", "1.81e-05", "43.4", "6.02e-05", "0 / 0", "통과"]],
      [2.8, 2.2, 2.4, 2.2, 2.4, 2.0, 1.6], aligns=["L", "C", "C", "C", "C", "C", "C"], size=8)
note("주: 같은 설정의 재실행(2026년 10월 6일)에서도 모든 배치가 통과하였다. 배치 처리의 속도 이득은 약 10% 수준이었다.")
figure("fig4_batch_gate.png", 14.5, "배치 추론 검증: 처리 시간과 한 건씩 계산 대비 최대 차이 (로그)")
para("분석에 쓰는 로짓은 fp16으로 계산되어 저장되어 있었다. 같은 200건을 fp32로 한 건씩 다시 계산하여 저장값과 비교하였다. 저장값은 한 건씩 재계산과 차이가 0이었으므로, 차이는 계산 정밀도에서 비롯된 것으로 판단하였다.")
caption_table("fp16 저장값과 fp32 계산값의 차이 (프로브 200건)")
table(["모델", "최대 차이", "평균(행별 최대 차이의 평균)", "중앙값", "최빈 뒤집힘"],
      [["EXAONE 3.5 2.4B", "7.72e-03", "1.65e-03", "1.25e-03", "1 / 200"],
       ["Kanana 1.5 2.1B", "1.48e-02", "1.58e-03", "4.00e-04", "1 / 200"]],
      [4.0, 2.8, 4.6, 2.6, 2.6], aligns=["L", "C", "C", "C", "C"])
note("주: 평균과 중앙값은 프로브마다 선택지별 차이의 최댓값을 구한 뒤 200건에 대해 계산하였다. 확률 전체의 평균 차이가 아니다.")
para("프로브당 평균 차이는 약 0.0016이었다. 주요 집단 대비인 EXAONE full−demo의 0.059와 비교하면 작은 수준이다. 이 점검은 200건 표본에 대한 것이므로, 정밀도 차이가 모든 결과에 같은 크기로 나타난다고 증명하지는 않는다. 다만 결론에 영향을 줄 수준은 아니라고 판단하였다.")

add_heading("6. 조건별 분석", 1)
add_heading("6.1 페르소나 없음 조건의 분포 재현 및 모드 붕괴", 2)
para("페르소나 없는 조건(C0)에서 255문항을 대상으로 모델 출력의 엔트로피를 실제 응답의 엔트로피와 비교하였다. EXAONE의 엔트로피 비율은 0.66, Kanana는 0.32였다. 두 모델 모두 실제보다 좁은 분포를 산출하였다. 셀 상관은 이론상 0 근처가 정상이므로, 셀 r은 지표의 타당성 점검으로만 해석한다.")
caption_table("C0 조건의 분포 재현 및 모드 붕괴 지표 (255문항)")
table(["모델", "평균 Wasserstein", "모델 엔트로피(bit)", "실제 엔트로피(bit)", "엔트로피 비율", "성별×연령 셀 r"],
      [["EXAONE 3.5 2.4B", "0.8475", "1.007", "1.533", "0.66", "−0.029"],
       ["Kanana 1.5 2.1B", "0.9230", "0.489", "1.533", "0.32", "0.001"]],
      [3.6, 2.6, 2.6, 2.6, 2.4, 2.4])

add_heading("6.2 본실험 29문항의 수준별 결과", 2)
para("본실험 29문항에 대해 네 수준(C0, adult, demo, full)과 인칭·화계 4조건으로 로짓을 계산하였다. 정규화 정확도는 모델 정확도를 같은 문항의 성별×연령 셀 최빈 정확도로 나눈 값이다. 페르소나 쌍은 같은 200명을 재사용하였고, 조건 간 차이의 95% 신뢰구간은 페르소나 단위 부트스트랩(2,000회)으로 구하였다.")
summ = pd.read_csv(BASE / "experiment/분석결과/수준별_요약.csv", encoding="utf-8-sig")
rows = []
for model, mname in [("exaone", "EXAONE 3.5 2.4B"), ("kanana", "Kanana 1.5 2.1B")]:
    for lev, lname in [("c0", "C0"), ("adult", "adult"), ("demo", "demo"), ("full", "full")]:
        sub = summ[(summ.model == model) & (summ.level == lev)]
        rows.append([mname, lname, f"{sub.acc.mean():.3f}", f"{sub.norm_acc.mean():.3f}",
                     f"{sub.wasserstein.mean():.3f}", f"{sub.entropy_ratio.mean():.3f}"])
caption_table("수준별 평균 (인칭·화계 4조건 평균, 29문항)")
table(["모델", "수준", "정확도", "정규화 정확도", "Wasserstein", "엔트로피 비율"], rows,
      [3.9, 2.0, 2.4, 2.9, 2.6, 2.7], aligns=["L", "L", "C", "C", "C", "C"])
note("주: 엔트로피 비율의 실제 엔트로피는 문항별로 다르므로 평균 비율은 참고값이다.")

pair = pd.read_csv(BASE / "experiment/분석결과/조건간_쌍체비교.csv", encoding="utf-8-sig")
caption_table("조건 간 쌍체 차이 (정규화 정확도 기준, 페르소나 부트스트랩 95% CI)")
rows = []
for _, r in pair.iterrows():
    rows.append(["EXAONE" if r.model == "exaone" else "Kanana",
                 r.contrast.replace("−", "−"),
                 f"{r.mean_diff:+.3f}", f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
                 "0 불포함" if (r.ci_low > 0 or r.ci_high < 0) else "0 포함"])
table(["모델", "대비", "평균 차이", "95% CI", "판정(탐색적)"], rows,
      [2.4, 3.0, 2.6, 4.0, 4.0], aligns=["L", "L", "C", "C", "L"])
figure("fig2_norm_accuracy.png", 14.5, "수준·인칭화계별 정규화 정확도 (점선: C0, 일점쇄선: adult, 빨간 점선: 셀 최빈 상한)")
figure("fig1_paired_contrasts.png", 13.5, "조건 간 쌍체 차이와 95% 신뢰구간")
para("EXAONE은 페르소나 6변수를 준 조건(full)이 인구통계 3변수 조건(demo)보다 정규화 정확도가 평균 0.059 높았다(95% CI [+0.013, +0.114]). C0와 비교해서도 0.052 높았다(95% CI [+0.001, +0.111]). Kanana는 세 조건 사이에 차이가 확인되지 않았다. 이 결과는 29개 문항의 평균에 근거한 탐색적 관측이다.")

add_heading("6.3 민감 문항의 편차 (H5)", 2)
para("민감 문항 9개에서 승인 답 확률의 모델 추정치와 실제 가중 확률의 차이(d)를 조건별로 계산하였다. 양수는 승인 쪽으로 치우친 것을 뜻한다. 이 사람 조건에서 편차가 작아지는 경향이 관측되었으나(Kanana full: 0.208에서 0.149), 검정은 실시하지 않았다. 해라체의 효과는 작았다. 따라서 H5는 탐색적으로만 보고하며, 해라체가 한국어로 부자연스러워 생긴 교란 가능성은 수용성 평정 이후에 판단한다.")

add_heading("7. 사전학습 오염 점검", 1)
para("모델이 KGSS 문항을 학습했을 가능성을 확인하기 위해 네 가지 프로브를 설계하였다. A는 로짓 분포의 재현, B는 원문 완성, C는 선택지 순서 정확도, E는 직접 회상 회피율을 측정한다. A와 B는 출처 명시, 미명시, 가짜 출처 조건을 두고 진짜 출처와 가짜 출처를 대조하였다. 표본은 140건이다.")
para("1차 점검에서 A/B의 일부 신호가 나타났으나, 선택지에 구조적 결측 코드가 실제 선택지처럼 섞여 제시되는 버그를 발견하였다. 이 버그는 A 프로브 351행 중 216행에 영향을 주었다. 수정 후 A, C, E를 재실행하였고, 그 결과를 표 8에 제시한다.")
caption_table("오염 점검 결과 (수정 후 재실행, n=140)")
table(["프로브", "비교", "결과", "해석"],
      [["A / EXAONE", "진짜 대 가짜 출처", "t=1.637, p=0.104", "유의하지 않음"],
       ["A / Kanana", "진짜 대 가짜 출처", "t=−0.253, p=0.801", "차이 없음(TOST p=0.029)"],
       ["B / Kanana", "진짜 대 가짜 출처", "9/10 결과: 가짜가 더 높음(p=0.006)", "1차 신호는 지시문의 정형화 효과로 설명됨"],
       ["C / EXAONE", "선택지 순서 정확일치", "29.3%(우연 기대 12.7%)", "통계검정 없는 보조 지표"],
       ["C / Kanana", "선택지 순서 정확일치", "5.7%(우연 이하)", "문제없음"],
       ["E / EXAONE", "회피율 진짜 대 가짜", "97.1% 대 95.7%, p=0.747", "차이 없음"],
       ["E / Kanana", "회피율 진짜 대 가짜", "72.9% 대 65.0%, p=0.197", "유의하지 않음"]],
      [2.4, 3.2, 4.4, 6.0], aligns=["L", "L", "L", "L"], size=8)
para("수정 후 재실행에서는 뚜렷한 오염 증거가 확인되지 않았다. 다만 C/EXAONE의 순서 정확일치율이 우연 기대치의 두 배를 넘으므로, 오염이 없다고 완전히 결론 내릴 수는 없다. 이 신호는 통계검정을 거치지 않은 보조 지표이다. 따라서 오염 가능성은 한계로 명시하며, 회차 확정은 팀의 결정 사항으로 남긴다.")

add_heading("8. 화계 파일럿", 1)
para("문항 4개와 인칭·화계 4조건으로 EXAONE과 Kanana의 암묵적 기댓값을 실제 분포와 비교하였다. 모든 판정은 방향성 관측이며 유의성 검정은 실시하지 않았다. EXAONE은 3인칭+해라체 조건이 기준 조건 대비 4문항 중 3문항에서 실제 분포에서 더 멀어졌다(Wasserstein 기준). 이는 가설과 다른 방향이다. Kanana는 문항마다 방향이 달라 일관된 패턴이 없었다.")

add_heading("9. 한계", 1)
para("본 장의 결과는 다음의 한계를 가진다. 첫째, 본실험 문항은 29개이고 표본은 200명의 페르소나이므로 검정력이 낮다. 둘째, 분석에 쓰인 로짓은 fp16 계산값이며, 정밀도 차이에 대한 점검은 200건 표본에 한정된다. 셋째, 6변수 페르소나 목록은 팀 확인 전의 임시 목록이다. 넷째, 인칭·화계 수준 수와 민감 기준은 수용성 평정과 문서화가 끝나야 확정된다. 따라서 본 장에서 제시한 조건 간 차이와 편차는 모두 탐색적 관측이며, 본실험에서 사전등록 후 검증하여야 한다.")

# ---------- 쪽번호 ----------
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run()
for tag, text in [("begin", None), (None, "PAGE"), ("end", None)]:
    if tag:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), tag)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    run._r.append(el)
set_font(run, size=9)

doc.save(OUT)
print("saved:", OUT)
