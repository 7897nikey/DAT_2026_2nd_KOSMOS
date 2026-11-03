"""중간보고서 수정본 생성 (python-docx). 표는 booktabs 스타일(굵은 상·하단선, 헤더 아래 얇은 선, 세로선 없음)."""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = Path(__file__).resolve().parent
FIG = BASE / "보고서_그림"
OUT = BASE / "중간보고서_수정본.docx"
FONT = "맑은 고딕"

doc = Document()

# ---------- 기본 설정 ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)


def set_font(style_or_run, size=None, bold=None, color=None):
    f = style_or_run.font
    f.name = FONT
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run, "element") else style_or_run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), FONT)
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color is not None:
        f.color.rgb = color


normal = doc.styles["Normal"]
set_font(normal, size=10.5)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size in [("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)]:
    st = doc.styles[name]
    set_font(st, size=size, bold=True, color=RGBColor(0, 0, 0))
    st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True

for name in ["List Bullet", "List Number"]:
    st = doc.styles[name]
    set_font(st, size=10.5)
    st.paragraph_format.line_spacing = 1.5
    st.paragraph_format.space_after = Pt(2)


def para(text, bold=False, size=None, align=None, space_after=None, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    r.italic = italic
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_font(r, size=10.5)
    return p


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size=8.5)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def caption_table(num, title):
    p = doc.add_paragraph()
    r = p.add_run(f"표 {num}  ")
    set_font(r, size=9.5, bold=True)
    r2 = p.add_run(title)
    set_font(r2, size=9.5, bold=False)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def caption_figure(num, title):
    p = doc.add_paragraph()
    r = p.add_run(f"그림 {num}  ")
    set_font(r, size=9.5, bold=True)
    r2 = p.add_run(title)
    set_font(r2, size=9.5, bold=False)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)


def figure(filename, width_cm, num, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(FIG / filename), width=Cm(width_cm))
    caption_figure(num, title)


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


def booktabs(headers, rows, widths_cm, aligns=None, font_size=9):
    """LaTeX booktabs 느낌의 표: 상단·하단 굵은 선(1.5pt), 헤더 아래 얇은 선(0.75pt), 세로선 없음."""
    ncols = len(headers)
    aligns = aligns or ["L"] + ["C"] * (ncols - 1)
    table = doc.add_table(rows=1 + len(rows), cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # 표 전체 테두리 제거
    tblPr = table._tbl.tblPr
    tb = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        tb.append(el)
    tblPr.append(tb)
    all_rows = [headers] + rows
    for i, row in enumerate(all_rows):
        for j, text in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.width = Cm(widths_cm[j])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = {"L": WD_ALIGN_PARAGRAPH.LEFT, "C": WD_ALIGN_PARAGRAPH.CENTER,
                           "R": WD_ALIGN_PARAGRAPH.RIGHT}[aligns[j]]
            r = p.add_run(str(text))
            set_font(r, size=font_size, bold=(i == 0))
            if i == 0:
                _border(cell, "top", 12)
                _border(cell, "bottom", 6)
            if i == len(all_rows) - 1:
                _border(cell, "bottom", 12)
    # 열 너비 고정
    for j, w in enumerate(widths_cm):
        table.columns[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def page_break():
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def fmt_sci(x):
    return f"{x:.2e}"


# ---------- 표지 ----------
for _ in range(5):
    para("", space_after=6)
para("DAT 8기 캡스톤 프로젝트 중간 보고서", bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
para("한국어 실리콘 샘플링의 타당성 검증", bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
para("LLM 응답 분포가 실제와 다를 때, 그 차이는 무엇 때문인가", size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
para("팀 명: KOSMOS", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para("팀 구성원: 김태우, 정주희(팀장), 허다교", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para("2026년 10월", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ---------- 국문 요약 ----------
doc.add_heading("국문 요약", level=1)
para("대규모 언어모델(LLM)에 인구통계 페르소나를 프롬프트로 부여해 여론조사 응답을 생성하는 실리콘 샘플링은 조사 비용을 낮출 수 있다는 기대를 받고 있다. 그러나 생성된 응답 분포가 실제 분포와 다르다는 보고가 함께 쌓여 왔다. 관측되는 사실은 \"다르다\" 하나이지만, 원인은 최소 네 가지로 나뉘며 선행연구는 이를 개별적으로 다루었다.")
para("본 연구는 한국종합사회조사(KGSS) 2003–2025 누적자료를 기준으로, 동일한 자료와 실험 틀에서 ① 정보 부족, ② 척도 표현(모드 붕괴), ③ 내용 편향(사회적 바람직성), ④ 형식 민감도(문구·선택지 순서)의 네 성분을 분해하는 것을 목표로 한다. 특히 ④는 인간 응답의 선택지 순서 효과를 A/B 분할표본으로 실측할 수 있어, 모델의 형식 민감도를 \"인간의 몇 배\"로 표현할 수 있는 유일한 성분이다.")
para("현재까지 2023년 회차를 주 자료로 문항 303개를 후보로 확보하였다. 본실험 문항은 29개로 확정하였다. 한국어 오픈웨이트 모델 두 계열(EXAONE 3.5, Kanana 1.5)에서 선택지 토큰화와 로짓 접근을 검증하였고, 배치 추론이 한 건씩 계산한 결과와 일치함을 확인하였다. 인구통계만으로 도달 가능한 개인 예측 정확도의 상한은 0.5385로 산정되었다. 예비 분석에서 EXAONE은 페르소나 6변수를 준 조건(full)의 정규화 정확도가 인구통계 3변수(demo)와 C0보다 높았으나, Kanana에서는 세 조건 사이에 차이가 없었다. 두 모델 모두 응답 분포가 실제보다 현저히 좁았다. 사전학습 오염 점검은 수정 후 재실행에서 뚜렷한 증거를 확인하지 못하였으나 완전한 배제는 불가능하므로 한계로 명시한다.")
para("주요어: 실리콘 샘플링, 모드 붕괴, 사회적 바람직성 편향, 형식 민감도, KGSS, 프롬프트 페르소나", size=10)
page_break()

# ---------- 목차 ----------
doc.add_heading("목 차", level=1)
toc = [
    ("제1장 서론", 0), ("제1절 연구 배경", 1), ("제2절 연구 목적 및 연구 질문", 1),
    ("제2장 데이터 소개 및 선행연구 분석", 0), ("제1절 선행연구 분석", 1), ("제2절 조사 및 분석 데이터 소개", 1),
    ("제3장 분석 진행 현황", 0), ("제1절 연구 모형 설계", 1), ("제2절 중간 분석 결과", 1),
    ("제4장 향후 계획", 0), ("제1절 남은 과제 및 보완 방향", 1), ("제2절 추진 일정", 1),
    ("참고문헌", 0),
]
for text, lvl in toc:
    p = para(("    " if lvl else "") + text, size=10.5, space_after=2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
para("쪽수는 최종 편집 시 자동 생성 목차로 채운다.", size=8.5, italic=True)
page_break()

# ---------- 제1장 ----------
doc.add_heading("제1장 서론", level=1)
doc.add_heading("제1절 연구 배경", level=2)
para("실리콘 샘플링(silicon sampling)은 LLM에 인구통계 특성을 주입해 가상의 설문 응답자를 만들고, 그 응답을 모아 실제 여론조사 분포를 근사하려는 방법이다. Argyle et al.(2023)은 LLM을 조건부 확률 계산기로 보고, 이 근사의 충실도를 알고리즘적 충실도(algorithmic fidelity)로 개념화하였다. 응답률 하락과 조사 비용 상승이라는 배경에서 이 방법은 빠르게 관심을 얻었다.")
para("그러나 생성된 분포는 실제와 체계적으로 다르다는 보고가 누적되어 있다. 첫째, 모드 붕괴(mode collapse)는 응답이 최빈값에 과도하게 몰려 분산이 비현실적으로 작아지는 현상이다(Li & Conrad, 2026; Heath & Alexander, 2026). 둘째, 사회적 바람직성 편향은 민감한 문항에서 실제 분포보다 승인되는 답으로 응답이 이동하는 현상으로, Sun et al.(2024)은 이를 harmlessness bias로 명명하였다. 두 현상은 서로 다른 연구 계보에서 개별적으로 다루어져 왔으며, 하나의 실험 틀에서 동시에 측정한 사례는 확인되지 않았다.")
para("또한 이 검증의 대부분은 영어 프롬프트와 미국 자료(ANES 등)에 집중되어 있다. 한국어에서 같은 결론이 성립하는지는 알려져 있지 않다. 본 연구는 이 공백을 한국어와 한국 인구집단 자료로 메우는 것을 출발점으로 한다.")
doc.add_heading("제2절 연구 목적 및 연구 질문", level=2)
para("본 연구의 중심 질문은 다음과 같다.")
para("\"LLM이 만든 응답 분포가 실제 분포와 다를 때, 그 차이는 무엇 때문인가?\"", bold=True)
para("관측되는 차이는 하나이나 원인은 최소 네 성분으로 나뉜다고 본다. 이 성분들을 동일한 자료와 실험 틀에서 분해하는 것이 목적이다. 연구 질문은 다음과 같이 구성한다.")
bullet("RQ1: 프롬프트 기반 페르소나 부여만으로 한국 인구집단의 응답 분포를 어디까지 재현하는가?")
bullet("RQ2: 집단 수준 재현이 성립할 때, 개인 수준의 응답 예측도 함께 성립하는가? 집단 분포를 맞추는 것과 개인을 맞히는 것은 목적이 다르며, 후자는 전자의 자동 결과가 아니다.")
bullet("RQ3: 차이를 만드는 네 성분(정보 부족, 척도 표현, 내용 편향, 형식 민감도) 중 어느 것이 얼마나 기여하는가?")
bullet("RQ4: 선택지 배열 순서가 모델 응답에 인간과 같은 방향의 효과를 주는가? (본 실험 포함 여부는 팀 결정 사항이다.)")
para("네 성분과 측정·개입 방법은 표 3-1에 정리하였다. 이 중 ④ 형식 민감도가 본 연구의 고유 기여이다. 인간 기준선이 확보된 유일한 성분이므로, 모델의 민감도를 \"많다/적다\"가 아니라 \"인간의 몇 배\"로 말할 수 있다.")

# ---------- 제2장 ----------
doc.add_heading("제2장 데이터 소개 및 선행연구 분석", level=1)
doc.add_heading("제1절 선행연구 분석", level=2)
para("본 절에서는 실리콘 샘플링과 관련된 주요 선행연구를 정리하고, 각 연구가 본 연구의 네 성분 중 어디에 대응하는지 표 2-1에 나타냈다. 확립된 결과와 저자들의 해석은 구분하여 기술하였다.")
caption_table("2-1", "주요 선행연구와 본 연구의 관계")
booktabs(
    ["연구", "다루는 성분", "보고된 결과(저자 보고 기준)", "본 연구와의 관계"],
    [
        ["Argyle et al. (2023)", "기반 개념", "LLM을 조건부 확률 계산기로 보고 알고리즘적 충실도 개념을 제시", "로짓 접근 설계의 근거"],
        ["Aher et al. (2023)", "모델 규모", "정교한 모델일수록 인간 패턴을 잘 재현하나, 일부 결과는 왜곡", "모델 계열·크기 축의 근거"],
        ["Li & Conrad (2026)", "① 정보 부족, ②", "인구통계만으로 페르소나를 구성하면 모드 붕괴가 발생", "인구통계 단독 조건의 근거"],
        ["Heath & Alexander (2026)", "②", "텍스트 응답 후 임베딩으로 척도에 매핑(SSR)하여 KL 1.97→0.07 감소, 평균 편향은 잔존", "척도 표현 성분의 대조 조건, 2회차 시간적 일반화 검증 대상"],
        ["Sun et al. (2024)", "③", "민감 문항에서 분포 재현이 가장 저조(harmlessness bias)", "RSS 조건, 내용 편향 성분의 출발점"],
        ["Chapala et al. (2025)", "③", "5조건 비교에서 삼인칭 재구성만 일관되게 효과. 요인 교차 설계가 아니어서 효과 원인 분리 불가", "인칭·화계를 교차 설계로 분리하는 것이 본 연구의 차별점"],
        ["Kim, Park & Suh (2026)", "①, ③, 개인 수준", "2024 서울서베이(확률표본) 대비 C0/C1/C2 3단 조건 비교. 서사 페르소나가 인구통계 단독보다 분포 재현과 셀 상관을 모두 악화. 상용 모델에서만 성립하고 오픈웨이트에서는 셀 상관이 붕괴", "직접 경쟁 연구. 한국 맥락 확률표본 검증은 선점되었으므로 본 연구는 인칭×화계 조작, 형식 민감도 인간 기준선, 개인 수준 분석을 고유 영역으로 둔다"],
    ],
    [3.2, 2.3, 5.8, 4.7], aligns=["L", "C", "L", "L"], font_size=8.5,
)
note("주: 저자 보고 내용을 요약한 것이며, 재현 여부는 본 연구에서 검증하지 않은 사항이다. Argyle et al.의 발표 연도는 자료에 따라 2022/2023이 혼재하므로 최종본에서 통일한다.")
para("1. 본 연구의 고유 영역", bold=True)
para("서울 논문(Kim, Park & Suh, 2026)의 등장으로 \"한국어 맥락에서 실리콘 샘플링을 확률표본으로 처음 검증한다\"는 주장은 사용하지 않는다. 다음 네 영역은 해당 논문이 다루지 않아 본 연구의 고유 기여로 남는다.")
bullet("언어 유형적 조작: 한국어는 상대높임법이 인칭과 독립된 문법 범주로 실현되므로, 영어에서 구조적으로 어려운 인칭과 화계의 독립 조작이 가능하다. 서울 논문은 서사 페르소나의 부작용을 관찰했을 뿐 원인을 분리하는 조작은 하지 않았다.")
bullet("형식 민감도의 인간 기준선: 두 선행 계열 모두 선택지 순서의 인간 기준선을 갖추지 않았다. KGSS의 A/B 분할표본으로 이를 실측할 수 있다.")
bullet("개인 수준 분석: 서울 논문의 지표는 population 수준 TVD와 셀 수준 상관뿐이다. 본 연구는 개인 예측 정확도 층위를 추가하고 셀 최빈 상한 대비로 정규화한다.")
bullet("오염 점검의 삼각측량: 서울 논문은 모델에게 \"기억나는지\"를 묻는 자기보고 프로브 하나에 의존한다. 본 연구는 로짓 분포, 원문 완성, 선택지 순서, 직접 회상을 병행하고 출처 명시와 가짜 출처 대조를 둔다.")

doc.add_heading("제2절 조사 및 분석 데이터 소개", level=2)
para("1. 자료 개요", bold=True)
para("한국종합사회조사(KGSS) 2003–2025 누적자료를 성균관대학교 서베이리서치센터(SRC)에서 확보하였다. 자료는 전국 가구 거주 만 18세 이상 성인을 모집단으로 하는 대면면접 조사이며, 23,282명의 응답자와 3,491개 변수, 17개 회차로 구성된다. 회차는 2003–2014년 매년, 이후 격년 조사이며 2020년은 결번이다. 2016년부터는 분할표본(split-ballot) 설계가 운영되었다.")
para("회차마다 문항이 달라 여러 해를 통합하면 비교가 성립하지 않으므로, 본 연구는 2023년을 주 회차(표본 1,230명)로, 2021년을 보조 회차로 선정하였다. 2021년은 2023년과 같은 국가자긍심 문항에 A/B 분할표본이 있어 선택지 순서 효과를 검증하는 데 쓴다. 응답자 식별자가 회차마다 재부여되므로 개인 수준 분석은 단일 회차 내로 제한된다.")
para("2. 자료 구조상 주의할 점", bold=True)
bullet("결측 코드는 -1(비해당)과 -8(모름·무응답) 두 개뿐이다. -1은 전체 셀의 89.6%를 차지하며, 해당 회차 미조사와 건너뛰기 규칙에 의한 비해당이 혼재한다. 그 해에 -1이 아닌 값이 하나라도 있으면 조사된 문항이고, 그 안의 -1은 건너뛰기 비해당으로 판정한다.")
bullet("값 라벨의 8, 88, 9, 99는 통합 이전 회차의 잔여물이므로, 코드값이 아니라 실질 선택지의 연속 구간으로 무응답(DK)을 판정한다.")
bullet("FINALWT는 회차 내 정규화되어 있어 연도 통합 가중 분석이 불가능하다. 층·집락 변수가 공개되지 않아 설계효과를 반영할 수 없으며, 신뢰구간은 실제보다 좁게 추정된다.")
bullet("실제 조사에서 \"모르겠다\"는 선택지로 제시되지 않았고 자발적 응답만 기록되었다. 보기카드로 선택지가 동시에 제시되었다.")
para("3. 2023년 회차 진단", bold=True)
caption_table("2-2", "2023년 회차 진단 요약")
booktabs(
    ["항목", "값", "비고"],
    [
        ["조사된 변수", "526 / 3,491", "회차 내 -1이 아닌 값 기준"],
        ["주 문항 후보", "303", "응답률 90% 이상, 비해당 5% 이하, 선택지 2–7개"],
        ["조건부 문항", "69", "스킵 비해당 5% 초과"],
        ["셀 최빈응답 정확도", "0.5385", "성별×연령 셀 최빈 기준(전체 최빈 0.5293)"],
        ["인구통계 순이익", "중앙값 0 (평균 0.0135)", "303문항 중 152문항은 정확히 0, 최대 0.185"],
        ["η² 중앙값", "0.018", "문항 선정의 층화 축으로만 사용"],
        ["성별×연령 셀 최소 n", "49 (12셀)", "학력까지 교차하면 n≥20인 셀이 54.5%로 감소"],
    ],
    [4.5, 4.3, 7.2], aligns=["L", "C", "L"],
)
note("주: 인구통계 순이익 중앙값이 0이라는 것은 성별·연령을 알아도 개인 예측이 전체 최빈 대비 개선되지 않는 문항이 절반 이상임을 뜻한다. 이는 개인 수준 예측의 상한이 인구통계로 정해진다는 가설(RQ2)을 지지하는 관측이지, 그 자체로 입증된 인과가 아니다.")

# ---------- 제3장 ----------
doc.add_heading("제3장 분석 진행 현황", level=1)
doc.add_heading("제1절 연구 모형 설계", level=2)
para("1. 네 성분과 측정·개입", bold=True)
para("연구 질문의 네 성분을 측정 지표와 개입 조건에 대응시킨 모형은 표 3-1과 같다.")
caption_table("3-1", "연구 모형: 네 성분과 측정·개입")
booktabs(
    ["성분", "어긋나는 지점", "측정", "개입"],
    [
        ["① 정보 부족", "인구통계로 개인을 특정할 수 없음", "셀 최빈응답 상한 대비 개인 예측 정확도", "응답 이력 조건 추가"],
        ["② 척도 표현", "출력 분포가 최빈값에 집중(모드 붕괴)", "출력 엔트로피, 셀 내 분산 재현율", "텍스트 응답 후 SSR 매핑, 디코딩 온도"],
        ["③ 내용 편향", "민감 문항에서 승인되는 답으로 이동", "민감 문항 방향성 편차", "인칭 × 화계 변형"],
        ["④ 형식 민감도", "문구·배열에 과민 반응", "TV거리, 인간 기준선 대비 배율", "정순/역순 교차"],
    ],
    [2.8, 4.6, 4.6, 4.0], aligns=["L", "L", "L", "L"],
)
para("2. 실험 조건", bold=True)
bullet("대조군 4수준: 페르소나 없음(C0), \"한국 성인\"(adult), 성별+연령(demo), 6변수(full: SEX·AGE·EDUC·MARITAL·REGION·URBAN, 팀 확인 전 임시 목록).")
bullet("페르소나 표본: 2023년 응답자를 FINALWT로 복원추출(n=200, seed 42)하여 생성하였다. 개별 변수를 따로 뽑지 않고 응답자 조합을 통째로 재표집하므로 6변수 결합분포가 실제와 일치한다. 한 번 뽑아 고정한 뒤 demo와 full에 재사용하여 대응비교를 한다. 시험 실행에서 6개 변수의 주변분포 오차는 1.7–3.3%p로 표본 노이즈 수준이었다.")
bullet("인칭×화계: 인칭 축은 페르소나를 청자로 부르는지(당신)와 관찰 대상으로 서술하는지(이 사람)로 잡는다. 응답이 숫자 하나이므로 화자 인칭은 출력에 드러나지 않는다. 화계는 하십시오체와 해라체를 비교한다. 수준 수는 수용성 평정 결과로 확정할 예정이다.")
bullet("문항: 본실험 문항은 29개로 확정하였다(사실 10, 가치관 10, 민감 9). 민감 문항은 화계 조작 효과가 드러날 문항 위주로 고르고, 배터리당 1개만 포함하였다.")
para("3. 모델과 측정 체계", bold=True)
para("모델은 EXAONE 3.5(2.4B)와 Kanana 1.5(2.1B)를 사용한다. 7.8B와 8B 모델은 GPU 환경에서 같은 프롬프트로 추가 실행할 계획이다. 재학습과 파인튜닝은 하지 않으며, 프롬프트 조건과 선택지 확률만 다룬다. 응답 형식은 번호 응답과 prefill로 고정하였다. 한국어 척도 선택지는 20개 중 19–20개가 다중 토큰이어서 텍스트 생성 후 첫 토큰으로 판별할 수 없기 때문이다.")
para("분포 비교 지표는 Wasserstein 거리를 주 지표로 쓰고 JS divergence를 병기한다. 순서형 척도에서 Wasserstein이 순서 정보를 반영하기 때문이다. TV거리는 인간 기준선 대비 배율에만 쓴다. 튜닝에 쓰는 지표와 평가에 쓰는 지표는 분리한다. 개인 예측 정확도는 모델의 최빈 응답이 실제 응답과 같은지로 정의하고, 같은 문항의 셀(성별×연령) 최빈 정확도로 나누어 정규화한다.")

doc.add_heading("제2절 중간 분석 결과", level=2)
para("본 절은 현재까지의 관측을 기술한다. 통계적으로 확인된 결과와 탐색적 관측, 해석을 구분하였다. 소표본 결과는 방향성만 기술한다. 본 분석 결과(3.6–3.8)는 한 건씩 계산한 fp16 로짓을 사용하였다. 정밀도 차이의 영향은 3.4절에서 점검하였다.")

para("1. 인간 응답의 선택지 순서 효과 (성분 ④ 기준선)", bold=True)
para("KGSS 2021·2023의 A/B 분할표본 16쌍 중 13쌍이 선택지 배열이 역전되어 있었다. 배열 효과의 크기는 TV거리 0.02–0.05였다. 국가자긍심 문항 10개 중 9개에서 효과 방향이 일치하였고, 우연으로 이 정도 일치할 확률은 약 1.1%이다. 방향은 뒤에 제시된 선택지가 더 선택되는 쪽이었다. 이 방향의 이론적 예측은 자명하지 않다. 본 조사는 낭독과 보기카드를 혼합한 방식이어서 최신 효과와 초두 효과가 어느 쪽으로 작용할지 미리 정할 수 없다. 따라서 본 결과는 경험적 관측으로만 기술한다.")
para("모델 쪽에서는 13쌍의 역순 문항에 대해 모델 TV거리를 인간 TV거리로 나눈 배율(표 3-2, 그림 3-1)을 계산하였다. EXAONE은 중앙값 0.71로 인간보다 작았으나, 13쌍 중 4쌍(PROUDMIL16, PROUDPOL16, NUKPLT10, ELEFRAUD)에서는 인간보다 크게 흔들렸다. Kanana는 중앙값 0.24로 대부분 인간보다 작았고, 13쌍 중 5쌍(NUKPLT10, PROUDSSS16, PROUDPOL16, PROUDDEM16 등)에서 인간보다 컸다. 두 모델 모두 응답 확률의 질량은 0.9998 이상이었다. 이 결과는 쌍별 표본이 13개에 불과하므로 방향성 관측이다.")
caption_table("3-2", "선택지 순서 효과의 인간 기준선 대비 배율 (13쌍)")
booktabs(
    ["구분", "인간 TV거리(중앙값)", "모델 배율 중앙값", "배율 > 1인 쌍", "비고"],
    [
        ["EXAONE 3.5 2.4B", "0.036", "0.71", "4 / 13", "최대 15.3배(PROUDPOL16)"],
        ["Kanana 1.5 2.1B", "0.036", "0.24", "5 / 13", "최대 48.8배(PROUDPOL16)"],
    ],
    [3.4, 3.2, 3.0, 2.8, 4.6], aligns=["L", "C", "C", "C", "L"],
)
note("주: 인간 TV거리는 KGSS 2023 A/B 쌍의 순서 효과다. 모델 TV거리는 정순과 역순 확률분포의 차이다. 배율은 모델 TV ÷ 인간 TV이다.")
figure("fig3_order_ratio.png", 14.0, "3-1", "선택지 순서 효과의 모델 대 인간 배율 (로그 척도; 1은 인간과 같은 크기)")

para("2. 토큰화와 로짓 접근의 실현 가능성", bold=True)
bullet("EXAONE 3.5와 Kanana 1.5 모두에서 숫자 1–9가 단일 토큰이었다.")
bullet("앞 공백이 별도 토큰으로 분리되므로, 프롬프트를 \"정답: \"처럼 공백으로 끝내 prefill하는 것이 필수임을 확인하였다.")
bullet("첫 토큰 확률질량은 EXAONE에서 0.983–1.0, Kanana에서 0.9998–1.0이었다. 로짓에 직접 접근하여 선택지 분포를 읽을 수 있음을 뜻한다.")
bullet("Kanana의 C0 조건 255건을 GPU 종류와 라이브러리 버전을 같게 두고 런타임을 새로 받아 재실행했을 때 원소별 최대 절대차가 0이었다(2026-10-04). 같은 조건에서 로짓 결과가 재현됨을 확인하였다.")
bullet("추가 후보인 Qwen2.5-3B-Instruct는 토큰화와 로짓 점검을 통과하였다. gemma-4-e2b-it는 아키텍처 인식을 위해 transformers 버전을 따로 고정해야 하므로 보류하였다.")

para("3. 배치 추론의 검증 (실행 환경 점검)", bold=True)
para("한 건씩 계산하던 로짓 추출을 배치로 바꾸면 시간을 줄일 수 있는지 확인하였다. 배치 처리에서는 왼쪽 패딩과 위치 번호를 맞추고, 마지막 위치의 로짓만 어휘 사전 전체에 대해 계산하도록 바꾸었다. 이 방식은 배치 32에서 발생하던 메모리 부족(OOM)을 없앴다. 판정 기준은 한 건씩 결과와의 최대 절대차 1e-3 미만, 최빈 선택 뒤집힘 0건으로 사전에 정하였다.")
caption_table("3-3", "배치 추론 게이트 결과 (fp32, 프로브 200건, 편집기 세션 1차 실행)")
booktabs(
    ["설정", "EXAONE 시간(초)", "EXAONE 최대 차이", "Kanana 시간(초)", "Kanana 최대 차이", "최빈 뒤집힘", "판정"],
    [
        ["한 건씩(기준)", "46.0", "–", "45.6", "–", "–", "–"],
        ["배치 8", "41.8", "1.60e-05", "40.5", "2.76e-05", "0 / 0", "PASS"],
        ["배치 16", "46.1", "1.63e-05", "42.1", "3.38e-05", "0 / 0", "PASS"],
        ["배치 32", "47.2", "1.81e-05", "43.4", "6.02e-05", "0 / 0", "PASS"],
    ],
    [2.8, 2.2, 2.4, 2.2, 2.4, 2.0, 1.6], aligns=["L", "C", "C", "C", "C", "C", "C"], font_size=8.5,
)
note("주: 같은 설정의 재실행(2026-10-06, 편집기 세션)에서도 모든 배치가 PASS였다. 재실행의 시간은 EXAONE 61.0–62.5초, Kanana 56.2–58.6초로 GPU 상태에 따라 달랐다. 속도 이득은 약 10% 수준이며, 배치 8이 차이가 가장 작고 가장 빨랐다.")
figure("fig4_batch_gate.png", 15.0, "3-2", "배치 추론 게이트: 처리 시간과 한 건씩 대비 최대 차이(로그)")

para("4. 정밀도 점검 (fp16 저장값과 fp32 계산값)", bold=True)
para("분석에 쓰는 로짓은 fp16으로 계산되어 저장되어 있다. 같은 200건을 fp32로 한 건씩 다시 계산하여 저장값과 비교하였다. 저장값은 한 건씩 재계산과 차이가 0이었으므로 저장 과정에서 바뀐 것은 없고, 차이는 계산 정밀도에서 나온다.")
caption_table("3-4", "fp16 저장값과 fp32 계산값의 차이 (프로브 200건)")
booktabs(
    ["모델", "최대 차이", "평균(행별 최대 차이의 평균)", "중앙값", "최빈 뒤집힘"],
    [
        ["EXAONE 3.5 2.4B", "7.72e-03", "1.65e-03", "1.25e-03", "1 / 200"],
        ["Kanana 1.5 2.1B", "1.48e-02", "1.58e-03", "4.00e-04", "1 / 200"],
    ],
    [4.0, 2.8, 4.6, 2.6, 2.6], aligns=["L", "C", "C", "C", "C"],
)
note("주: 평균과 중앙값은 프로브마다 선택지별 차이의 최댓값을 구한 뒤 200건에 대해 계산한 값이다. 확률 전체의 평균 차이가 아니다.")
para("프로브당 평균 차이는 약 0.0016으로, 본 분석의 집단 차이(예: EXAONE full−demo 0.059)보다 훨씬 작다. 최빈 선택 뒤집힘은 두 모델 모두 200건 중 1건(0.5%)이다. 이 점검은 200건 표본이므로, 정밀도 차이가 모든 결과에 같은 크기로 나타난다는 것을 증명하지는 않는다. 그러나 본 연구의 주요 대비 크기와 비교하면 결론에 영향을 줄 수준은 아니라고 판단하였다.")

para("5. 모드 붕괴 점검 (C0, 페르소나 없음)", bold=True)
para("페르소나 없는 조건(C0)에서 255문항을 대상으로 모델 출력의 엔트로피를 실제 응답 엔트로피와 비교하였다.")
caption_table("3-5", "C0 조건의 분포 재현과 모드 붕괴 지표 (255문항)")
booktabs(
    ["모델", "평균 Wasserstein", "모델 엔트로피(bit)", "실제 엔트로피(bit)", "엔트로피 비율", "성별×연령 셀 r"],
    [
        ["EXAONE 3.5 2.4B", "0.8475", "1.007", "1.533", "0.66", "−0.029"],
        ["Kanana 1.5 2.1B", "0.9230", "0.489", "1.533", "0.32", "0.001"],
    ],
    [3.6, 2.6, 2.6, 2.6, 2.4, 2.4], aligns=["L", "C", "C", "C", "C", "C"],
)
note("주: 셀 r은 C0에서 셀별 구체 정보가 없으므로 이론상 0 근처가 정상이다(서울 논문 C0 실측 0.03과 비교). 이 값은 지표의 타당성 점검이며, 조건화 능력의 측정이 아니다. 엔트로피 비율은 두 모델 모두 실제보다 좁은 분포를 낸다는 관측이다.")

para("6. 본 분석 결과: 페르소나 조건과 정확도 (탐색적, 29문항)", bold=True)
para("본실험 29문항에 대해 네 수준(C0, adult, demo, full)과 인칭·화계 4조건으로 로짓을 계산하였다. 정확도는 모델의 최빈 응답이 페르소나의 실제 응답과 같은 비율이며, 정규화 정확도는 같은 문항의 성별×연령 셀 최빈 정확도로 나눈 값이다. 페르소나 쌍은 같은 200명을 재사용하고, 조건 간 차이는 페르소나 단위 부트스트랩(2,000회)으로 95% 신뢰구간을 구하였다.")
pair = pd.read_csv(BASE / "experiment/분석결과/조건간_쌍체비교.csv", encoding="utf-8-sig")
summ = pd.read_csv(BASE / "experiment/분석결과/수준별_요약.csv", encoding="utf-8-sig")
caption_table("3-6", "조건 수준별 평균 (인칭·화계 4조건 평균, 29문항)")
rows = []
for model, mname in [("exaone", "EXAONE 3.5 2.4B"), ("kanana", "Kanana 1.5 2.1B")]:
    for lev, lname in [("c0", "C0"), ("adult", "adult"), ("demo", "demo"), ("full", "full")]:
        sub = summ[(summ.model == model) & (summ.level == lev)]
        rows.append([mname, lname, f"{sub.acc.mean():.3f}", f"{sub.norm_acc.mean():.3f}",
                     f"{sub.wasserstein.mean():.3f}", f"{sub.entropy_ratio.mean():.3f}"])
booktabs(
    ["모델", "수준", "정확도", "정규화 정확도", "Wasserstein", "엔트로피 비율"],
    rows, [3.9, 2.0, 2.4, 2.9, 2.6, 2.7], aligns=["L", "L", "C", "C", "C", "C"],
)
note("주: 엔트로피 비율은 모델 엔트로피를 실제 응답의 엔트로피로 나눈 값이다. 실제 엔트로피는 문항별로 다르므로 평균 비율은 참고값이다. C0와 adult는 인칭·화계가 없어 한 조건으로 계산하였다.")

caption_table("3-7", "조건 간 쌍체 차이 (정규화 정확도 기준, 페르소나 부트스트랩 95% CI)")
name_map = {"full − demo": "full − demo", "demo − c0": "demo − C0", "full − c0": "full − C0"}
rows = []
for _, r in pair.iterrows():
    rows.append([
        "EXAONE" if r.model == "exaone" else "Kanana",
        name_map.get(r.contrast.replace("−", "−"), r.contrast),
        f"{r.mean_diff:+.3f}", f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
        "CI가 0을 벗어남" if (r.ci_low > 0 or r.ci_high < 0) else "CI가 0을 포함",
    ])
booktabs(
    ["모델", "대비", "평균 차이", "95% CI", "판정(탐색적)"],
    rows, [2.4, 3.0, 2.6, 4.0, 4.0], aligns=["L", "L", "C", "C", "L"],
)
note("주: 문항 29개의 평균 차이이며, 신뢰구간은 페르소나 단위 재표집으로 구했다. 문항 단위 결론은 탐색적이다. 검정력 분석(d_z ≈ 0.20, n=200)에 따르면 이 표본은 작은 효과를 찾기에 충분하지 않을 수 있다.")
figure("fig2_norm_accuracy.png", 15.0, "3-3", "수준·인칭화계별 정규화 정확도 (점선: C0, 일점쇄선: adult, 빨간 점선: 셀 최빈 상한 1.0)")
figure("fig1_paired_contrasts.png", 14.0, "3-4", "조건 간 쌍체 차이와 95% CI (페르소나 부트스트랩)")
para("결과를 요약하면 다음과 같다. EXAONE은 인구통계 6변수를 주었을 때(full) 인구통계 3변수(demo)보다 정규화 정확도가 평균 0.059 높았고(95% CI [+0.013, +0.114]), C0보다도 0.052 높았다(95% CI [+0.001, +0.111]). Kanana는 세 조건 사이에 차이가 없었다. 엔트로피 비율은 EXAONE이 0.71–0.79, Kanana가 0.34–0.48 범위에 머물러, 페르소나 조건에 따른 변화가 작았다. 따라서 모드 붕괴가 페르소나 때문이라기보다 모델 성질일 가능성이 있다(Q2 대응, 탐색적).")

para("7. 민감 문항의 편차 (H5, 탐색적)", bold=True)
para("민감 문항 9개의 승인 답 확률에서 모델 확률을 실제 가중 확률로 뺀 편차(d)를 조건별로 계산하였다. 양수는 승인 쪽으로 치우친 것이다.")
h5 = pd.read_csv(BASE / "experiment/분석결과/H5_민감편차.csv", encoding="utf-8-sig")
h5m = h5.groupby(["model", "level", "cond"]).d.mean().reset_index()
caption_table("3-8", "민감 문항 편차 d의 조건별 평균 (승인 답 확률 차이)")
rows = []
for model, mname in [("exaone", "EXAONE"), ("kanana", "Kanana")]:
    for lev, lname in [("demo", "demo"), ("full", "full")]:
        vals = []
        for c in ["당신·하십시오체", "당신·해라체", "이사람·하십시오체", "이사람·해라체"]:
            v = h5m[(h5m.model == model) & (h5m.level == lev) & (h5m.cond == c)].d
            vals.append(f"{v.iloc[0]:+.3f}" if len(v) else "–")
        rows.append([mname, lname] + vals)
booktabs(
    ["모델", "수준", "당신·하십시오", "당신·해라", "이 사람·하십시오", "이 사람·해라"],
    rows, [2.2, 1.8, 3.1, 3.1, 3.4, 3.0], aligns=["L", "L", "C", "C", "C", "C"],
)
note("주: 평균은 민감 문항 전체(9개)에 대한 값이다. 이 사람 조건에서 편차가 작아지는 경향이 관측되었으나(특히 Kanana full: 0.208 → 0.149), 검정은 하지 않았고 문항 9개의 평균이므로 탐색적이다. 해라체의 효과는 작다. 해라체 자체가 어색하다는 교란 가능성을 남겨 두며, 해석은 수용성 평정 이후로 미룬다.")

para("8. 사전학습 오염 점검", bold=True)
para("모델이 KGSS 문항을 이미 학습했을 가능성을 점검하기 위해 네 가지 프로브를 설계하였다. A는 로짓 분포의 실제 분포 재현, B는 원문 완성, C는 선택지 순서 정확도, E는 직접 회상 회피율이다. A와 B는 출처 명시, 미명시, 가짜 출처(존재하지 않는 조사명)의 3조건을 두고, 진짜 출처와 가짜 출처를 대조하여 프레이밍 효과와 기억을 구분하였다.")
para("점검 과정에서 두 번의 수정이 있었다. 9월 9–10일 1차 결과는 A/B의 진짜 대 미명시 차이에서 유의한 신호를 보였으나, 가짜 출처 대조에서 B/Kanana의 신호는 반대 방향으로 뒤집혔다. 이후 9월 22일에는 선택지에 구조적 결측 코드(DK, 비해당)가 실제 선택지처럼 섞여 제시된 버그를 발견하여 A, C, E를 재실행하였다. 버그는 A 프로브 351행 중 216행에 영향을 주었다. 표 3-9는 수정 후 재실행 값이다.")
caption_table("3-9", "오염 점검 결과 (2026-09-22 재실행, n=140)")
booktabs(
    ["프로브", "비교", "결과", "해석"],
    [
        ["A / EXAONE", "진짜 대 가짜 출처", "t=1.637, p=0.104", "유의하지 않음. 1차에서 남았던 신호가 버그 수정 후 소실"],
        ["A / Kanana", "진짜 대 가짜 출처", "t=−0.253, p=0.801", "차이 없음. TOST로 동등성 확인(p=0.029)"],
        ["B / Kanana", "진짜 대 가짜 출처", "9/10 결과: 가짜가 더 높음(p=0.006)", "1차 신호는 지시문의 정형화 효과로 설명되며 기억 증거가 아님"],
        ["C / EXAONE", "선택지 순서 정확일치", "29.3% (우연 기대 12.7%)", "통계검정 없는 보조지표. 유일하게 남은 걱정거리"],
        ["C / Kanana", "선택지 순서 정확일치", "5.7% (우연 이하)", "문제없음"],
        ["E / EXAONE", "회피율 진짜 대 가짜", "97.1% 대 95.7%, p=0.747", "차이 없음"],
        ["E / Kanana", "회피율 진짜 대 가짜", "72.9% 대 65.0%, p=0.197", "유의하지 않음"],
    ],
    [2.4, 3.2, 4.4, 6.0], aligns=["L", "L", "L", "L"], font_size=8.5,
)
note("주: 결론은 다음과 같다. 수정된 재실행에서 뚜렷한 오염 증거는 확인되지 않았다. 다만 C/EXAONE의 순서 정확일치율이 우연 기대치의 두 배를 넘으므로, 오염이 없다고 완전히 결론 내릴 수는 없다. 이 신호는 통계검정을 거치지 않은 보조 지표이다.")
para("회차 선택과 관련하여 한 가지를 짚는다. 2021년 회차는 2023년과 같은 공개 자료에 속하므로 노출 위험이 같다. 따라서 회차를 바꾼다고 오염 위험이 줄지는 않는다. 2025년 회차는 두 모델의 공개 이후 조사되었을 가능성이 높아 노출 구간 밖일 여지가 있다. 다만 모델의 학습 cutoff는 공식 공개되지 않아 \"배제할 수 없다\"까지만 말할 수 있으며, 2025년 문항 구성은 확인하지 않았다.")

para("9. 문항 태깅과 문항유형 병합", bold=True)
para("문항 303개를 세 명이 독립적으로 문항유형(사실, 가치관, 차별)과 민감 여부로 태깅하였다. 문항유형의 일치도는 Fleiss κ 0.775로 양호했다. 민감 여부의 일치도는 Fleiss κ 0.424로 낮았으며, 판정이 갈린 문항은 67개였다.")
para("문항유형은 다수결로 병합하였고, 3명의 판정이 모두 다른 동률 문항은 없었다. 불일치 52문항은 모두 2대 1로 결정되었다. 병합 결과는 가치관 194, 사실 92, 차별 17이다. 민감 여부는 태깅 기준(사실 문항은 공란)에 맞춰 정리하였다. 이 과정에서 사실 문항 10개에 민감 N이 기록되어 있었으나 기준에 따라 공란으로 처리하였다. 이 처리는 연구팀의 판단이며 확정 전 재확인이 필요하다. 그 결과 민감 문항은 47개(가치관 30, 차별 17)가 되었다.")
caption_table("3-10", "문항유형과 민감 여부의 교차 (병합 후)")
booktabs(
    ["문항유형", "민감 Y", "민감 N", "공란", "합계"],
    [
        ["가치관", "30", "164", "0", "194"],
        ["차별", "17", "0", "0", "17"],
        ["사실", "0", "0", "92", "92"],
        ["합계", "47", "164", "92", "303"],
    ],
    [4.0, 3.0, 3.0, 3.0, 3.0], aligns=["L", "C", "C", "C", "C"],
)

para("10. 화계 파일럿 (방향성만 기술)", bold=True)
para("문항 4개, 조건 4개(당신·하십시오체, 당신·해라체, 이 사람·하십시오체, 이 사람·해라체)로 EXAONE과 Kanana의 암묵적 기댓값을 실제 분포와 비교하였다. 모든 판정은 방향성 관측이며 유의성 검정은 하지 않았다.")
bullet("EXAONE: 3인칭+해라체(D)는 기준 조건(A) 대비 4문항 중 3문항에서 실제 분포에서 더 멀어졌다(Wasserstein 기준). 한 문항(IMMECON)에서만 가까워졌다.")
bullet("Kanana: 방향이 문항마다 달라 일관된 패턴이 없었다.")
para("이 결과는 가설(3인칭+해라체일수록 사회적 바람직성이 줄어든다)과 다른 방향이다. 다만 팀 내에서 3인칭+해라체가 한국어로 어색해서 그 자체가 교란이 되었을 수 있다는 지적이 있었다. 본 연구는 이 조건을 바로 제외하지 않고, 먼저 수용성 평정으로 판단하기로 하였다. 그 결과에 따라 유지, 수정, 제외를 정한다.")

para("11. 선행 파일럿 (2021년, 팀원 실행)", bold=True)
para("팀원이 2021년 응답자 1,203명의 프로필을 대상으로 Direct 방식과 RSS 방식을 비교하였다(GPT-5.6 Luna, 18,045건, 오류 0). 모델 응답은 인간에 비해 특정 선택지로 크게 쏠렸고, 모델의 연령 Cramér's V는 0.328로 인간(0.087)의 약 3.8배였다. Direct와 RSS의 Wasserstein 거리는 거의 같았다(0.641 대 0.640). 또한 top_logprobs 상한 때문에 선택지 토큰이 누락되는 문제가 확인되어, 오픈웨이트로 전환하는 근거가 되었다.")

# ---------- 제4장 ----------
doc.add_heading("제4장 향후 계획", level=1)
doc.add_heading("제1절 남은 과제 및 보완 방향", level=2)
para("운영진 피드백 여섯 가지(Q1–Q6)에 대한 현재 대응 상태와 다음 과제는 표 4-1과 같다.")
caption_table("4-1", "운영진 지적 대응 현황")
booktabs(
    ["지적", "내용", "현재 상태", "다음 과제"],
    [
        ["Q1", "Direct와 RSS 결과가 같다면 페르소나를 주입하는 이유", "미완", "RSS 프로필 중 실제 데이터에 드문 조합(하위 10%)만 골라 Direct와 비교"],
        ["Q2", "최빈 쏠림이 페르소나인지 모델 성질인지 불분명", "C0 및 수준별 분석 완료(탐색적)", "엔트로피 비율이 조건에 따라 거의 변하지 않아 모델 성질 가능성. 본실험에서 확인"],
        ["Q3", "η² 0.018은 누구도 맞히지 못한다는 뜻이므로 상한 대비 정규화 필요", "29문항 정규화 정확도 계산 완료", "파일럿 5문항 확대 및 본실험 결과에 적용"],
        ["Q4", "오염이 확인될 때 대안 경로가 없음", "수정 후 재실행 완료, 판정 보류", "2025 회차 문항 구성 확인 후 대안 경로 필요성 결정"],
        ["Q5", "선택지 토큰화 미확인", "완료", "로짓 누출률(선택지 밖 확률질량) 보고"],
        ["Q6", "문항 선정 기준 부재", "태깅·병합 완료, 본실험 29문항 확정", "민감 문항 선정 기준을 Seoul 논문 Table 3 방식으로 문서화"],
    ],
    [1.3, 4.6, 4.3, 5.8], aligns=["C", "L", "L", "L"], font_size=8.5,
)
para("미결 정 사항 (팀 결정 필요)", bold=True)
bullet("회차 확정: 현재 2023 유지. 오염 가능성이 완전히 배제되지 않으므로 2025 대안을 검토할지 결정 필요.")
bullet("RQ4(순서 효과)의 본 실험 포함 여부: 운영진 권고는 포함.")
bullet("상용 모델 유지 여부: 본 실험은 오픈웨이트 2×2를 기본으로 하되, 상용 모델 비교를 부차 분석으로 둘지 결정 필요.")
bullet("6변수 페르소나 목록: 현재 SEX, AGE, EDUC, MARITAL, REGION, URBAN으로 임시 구성하였으며 팀 확인이 필요하다.")
bullet("민감 판정 불일치 67문항 재논의 및 민감 기준 문서화(κ 0.424 보완).")
bullet("디코딩 온도를 실험 요인으로 포함할지 여부(선행연구 한계에 대한 대응).")

doc.add_heading("제2절 추진 일정", level=2)
para("아래 일정은 작업의 선후 관계를 기준으로 정리하였다. 기간은 팀 회의에서 확정한다.")
caption_table("4-2", "추진 단계와 선행 조건")
booktabs(
    ["단계", "내용", "선행 조건", "상태"],
    [
        ["1", "인칭×화계 수용성 평정(4조건 × 4문항, 5점 척도)", "없음", "미착수"],
        ["2", "민감 문항 선정 기준 문서화 및 불일치 재논의", "없음", "미착수"],
        ["3", "문항 30개 확정 → 29개 확정(유형별 10·10·9)", "단계 1, 2", "29문항 확정"],
        ["4", "6변수 페르소나 목록 확정 및 표본 재생성(필요 시)", "팀 결정", "임시 목록으로 진행"],
        ["5", "대조군 수준별 실행과 Q1–Q3 분석", "단계 3, 4", "실행 및 탐색 분석 완료, Q1 미완"],
        ["6", "사전등록 및 본 실험", "단계 1–5, 회차·RQ4 결정", "[기간 확정 필요]"],
    ],
    [1.4, 7.4, 3.6, 3.6], aligns=["C", "L", "L", "L"], font_size=8.5,
)

# ---------- 참고문헌 ----------
doc.add_heading("참고문헌", level=1)
doc.add_heading("국내문헌", level=2)
para("한국종합사회조사(KGSS) 2003–2025 누적자료 및 코드북 v5. 성균관대학교 서베이리서치센터.")
para("[국내 문헌 추가 예정] 한국어 LLM 편향 및 문화 정렬 관련 국내 연구는 아직 조사하지 않았다.")
doc.add_heading("국외문헌", level=2)
for ref in [
    "Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023). Using large language models to simulate multiple humans and replicate human subject studies. In Proceedings of the 40th International Conference on Machine Learning (ICML). arXiv:2208.10264.",
    "Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. Political Analysis.",
    "Chapala, S., Mironov, M., & Deng, S. (2025). Mitigating social desirability bias in random silicon sampling. arXiv:2512.22725.",
    "Heath, O., & Alexander, R. (2026). Correcting mode collapse in silicon sampling with semantic similarity rating. arXiv:2607.28550.",
    "Kim, Park, & Suh (2026). Silicon sampling in Seoul: Conditioning ablations for LLM survey simulation against a probability sample.",
    "Li, S., & Conrad, F. (2026). Persona-based simulation of human opinion at population scale. arXiv:2603.27056.",
    "Sun, T., et al. (2024). Random silicon sampling: Simulating human sub-population opinion using a large language model based on group-level demographic information. arXiv:2402.18144.",
    "Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., & Larson, J. M. (2024). Synthetic replacements for human survey data? The perils of large language models. Political Analysis, 32(4), 401–416. https://doi.org/10.1017/pan.2024.5",
    "[서지 확인 필요: 저자·발행 연도] Evaluating silicon sampling: LLM accuracy in simulating public opinion on facial recognition technology. AHFE 학회 논문집(Open Access). https://openaccess.cms-conferences.org/publications/book/978-1-964867-73-1/article/978-1-964867-73-1_45",
]:
    p = para(ref, size=9.5)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.line_spacing = 1.3

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
