import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

# ---------------- Paths & Defaults ----------------
DATA_FILE   = Path("bodylog_plus.json")
CONFIG_FILE = Path("bodylog_plus_config.json")
PROFILE_FILE= Path("bodylog_plus_profile.json")

DEFAULT_CONFIG = {
    "metrics": ["bp", "hr", "temp", "sugar"],  # 기본 노출 항목
    "thresholds": {
        "bp_sys_hi": 140, "bp_dia_hi": 90,
        "bp_sys_very": 180, "bp_dia_very": 120,
        "hr_lo": 50, "hr_hi": 120,
        "temp_hi": 38.5,
        "sugar_hi": 180, "sugar_very": 240, "sugar_lo": 60,
        "spo2_lo": 92,
        "rr_lo": 10, "rr_hi": 24,
    }
}

METRIC_META = {
    "bp":     {"label": "혈압(수축/이완)", "type": "text",  "placeholder": "120/80", "unit": "mmHg"},
    "hr":     {"label": "심박수(bpm)",     "type": "int",   "unit": "bpm"},
    "temp":   {"label": "체온(°C)",        "type": "float", "step": 0.1, "unit": "°C"},
    "sugar":  {"label": "혈당(mg/dL)",     "type": "float", "step": 0.1, "unit": "mg/dL"},
    "spo2":   {"label": "SpO₂(%)",         "type": "int",   "unit": "%"},
    "rr":     {"label": "호흡수(RR)",      "type": "int",   "unit": "/min"},
    "weight": {"label": "체중(kg)",        "type": "float", "step": 0.1, "unit": "kg"},
    "waist":  {"label": "허리둘레(cm)",    "type": "float", "step": 0.1, "unit": "cm"},
    "bmi":    {"label": "BMI(kg/m²)",      "type": "float", "step": 0.1, "unit": "kg/m²"},
}

# 그래프 표시용 라벨/단위 (혈압은 분리지표 사용)
PLOT_META = {
    "hr": ("심박수(bpm)", "bpm"), "temp": ("체온(°C)", "°C"), "sugar": ("혈당(mg/dL)", "mg/dL"),
    "spo2": ("SpO₂(%)", "%"), "rr": ("호흡수(/min)", "/min"),
    "weight": ("체중(kg)", "kg"), "bmi": ("BMI(kg/m²)", "kg/m²"),
    "bp_sys": ("수축기(mmHg)", "mmHg"), "bp_dia": ("이완기(mmHg)", "mmHg"),
}

def make_plot_options(active_metrics: List[str]) -> List[str]:
    opts = []
    for m in active_metrics:
        if m == "bp":
            opts += ["bp_sys", "bp_dia"]
        elif m in PLOT_META:
            opts.append(m)
    seen = set()
    return [x for x in opts if not (x in seen or seen.add(x))]

# ---------------- IO helpers ----------------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- Logic helpers ----------------
def parse_bp(bp_str: str) -> Tuple[int | None, int | None]:
    try:
        s, d = bp_str.split("/")
        return int(s.strip()), int(d.strip())
    except Exception:
        return None, None

def abnormal_flags(row: Dict[str, Any], thr: Dict[str, Any]) -> str:
    flags: List[str] = []
    # 혈압
    if "bp" in row:
        s, d = parse_bp(str(row.get("bp", "")))
        if s and d:
            if s >= thr["bp_sys_very"] or d >= thr["bp_dia_very"]:
                flags.append("혈압 매우 높음")
            elif s >= thr["bp_sys_hi"] or d >= thr["bp_dia_hi"]:
                flags.append("혈압 높음")
    # 심박
    if isinstance(row.get("hr"), (int, float)) and (row["hr"] < thr["hr_lo"] or row["hr"] > thr["hr_hi"]):
        flags.append("심박 비정상")
    # 체온
    if isinstance(row.get("temp"), (int, float)) and row["temp"] >= thr["temp_hi"]:
        flags.append("고열")
    # 혈당
    if isinstance(row.get("sugar"), (int, float)):
        if row["sugar"] >= thr["sugar_very"] or row["sugar"] <= thr["sugar_lo"]:
            flags.append("혈당 위험")
        elif row["sugar"] >= thr["sugar_hi"]:
            flags.append("혈당 높음")
    # SpO2
    if isinstance(row.get("spo2"), (int, float)) and row["spo2"] < thr["spo2_lo"]:
        flags.append("저산소")
    # 호흡수
    if isinstance(row.get("rr"), (int, float)) and (row["rr"] < thr["rr_lo"] or row["rr"] > thr["rr_hi"]):
        flags.append("호흡수 이상")
    return ", ".join(flags)

# (선택) 경고음
def make_beep_wav(seconds=0.35, freq=880, rate=44100):
    try:
        t = np.linspace(0, seconds, int(rate*seconds), False)
        tone = 0.5*np.sin(2*np.pi*freq*t)
        audio = np.int16(tone*32767)
        from scipy.io.wavfile import write as wav_write
        bio = BytesIO()
        wav_write(bio, rate, audio)
        bio.seek(0)
        return bio
    except Exception:
        return None

# ---------------- UI: Global ----------------
st.set_page_config(page_title="📝 바디로그 PLUS", page_icon="📝", layout="wide")

# ---- 시간 입력 박스 전용 스타일 ----
st.markdown("""
<style>
/* .time-narrow 래퍼 안의 input만 타겟팅 */
.time-narrow input {
  width: 80px !important;      /* ← 더 줄이고 싶으면 숫자만 변경 (예: 70~100) */
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

# 테마(간단)
with st.sidebar:
    st.markdown("### 🎨 화면 테마")
    theme = st.selectbox("테마", ["차분한 블루", "따뜻한 베이지", "다크 모드", "우드 모드"], index=0)
palettes = {
    "차분한 블루":  ("#eef6fb", "#ffffff", "#90caf9", "#0f172a"),
    "따뜻한 베이지":("#fff8e7", "#ffffff", "#e2c799", "#1f2937"),
    "다크 모드":    ("#1e1e1e", "#2a2a2a", "#555555", "#e5e7eb"),
    "우드 모드":    ("#f5f2e7", "#ffffff", "#9bbf87", "#3d3322"),
}
BG, CARD, BORDER, TEXT = palettes[theme]

st.markdown(f"""
<style>
.stApp {{ background-color:{BG}; color:{TEXT}; }}
.stTextInput, .stNumberInput, .stDateInput, .stTimeInput,
.stSelectbox, .stTextArea, .stSlider, .stMultiSelect {{
  border:2px solid {BORDER}!important; border-radius:12px!important;
  padding:8px!important; background:{CARD}!important;
}}
.stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input,
.stTextArea textarea {{ background:{CARD}!important; color:{TEXT}!important; }}
.stButton button {{
  border-radius:10px!important; border:1px solid {BORDER}!important;
  background:{CARD}!important; color:{TEXT}!important; padding:.6rem 1rem!important;
}}
/* 글자 크게 */
label, .stMarkdown p {{ font-size:20px!important; font-weight:600; }}
.stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input, .stTextArea textarea {{ font-size:22px!important; }}
div[data-baseweb="select"] * {{ font-size:20px!important; }}
.stDataFrame, .stDataFrame * {{ font-size:18px!important; }}
</style>
""", unsafe_allow_html=True)

st.title("📝 바디로그 PLUS — 선택형 지표/경고/그래프/PDF 리포트")

# ---------------- Load DB & migrate IDs (하나만) ----------------
DB       = load_json(DATA_FILE,   {"entries": []})
CFG      = load_json(CONFIG_FILE, DEFAULT_CONFIG)
PROFILE  = load_json(PROFILE_FILE, {"height_cm": None})

from uuid import uuid4
def migrate_ids(db: dict) -> int:
    changed = 0
    for e in db.get("entries", []):
        if "id" not in e:
            e["id"] = uuid4().hex
            changed += 1
    if changed:
        save_json(DATA_FILE, db)
    return changed

migrated = migrate_ids(DB)
if migrated:
    st.toast(f"기존 기록 {migrated}건에 ID 부여 완료")

# ---------------- Sidebar: 설정 ----------------
with st.sidebar:
    st.subheader("⚙️ 추적 지표 설정")
    default_checked = set(CFG.get("metrics", []))

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**기본 바이탈**")
        bp_on    = st.checkbox("혈압(수축/이완) — mmHg", value=("bp" in default_checked))
        hr_on    = st.checkbox("심박수(bpm)", value=("hr" in default_checked))
        temp_on  = st.checkbox("체온(°C)", value=("temp" in default_checked))
        sugar_on = st.checkbox("혈당(mg/dL)", value=("sugar" in default_checked))
    with colB:
        st.markdown("**옵션**")
        spo2_on  = st.checkbox("SpO₂(%)", value=("spo2" in default_checked))
        rr_on    = st.checkbox("호흡수(RR /min)", value=("rr" in default_checked))
        weight_on= st.checkbox("체중(kg)", value=("weight" in default_checked))
        waist_on = st.checkbox("허리둘레(cm)", value=("waist" in default_checked))
        bmi_on   = st.checkbox("BMI(kg/m²)", value=("bmi" in default_checked))

    if st.button("저장(지표 설정)"):
        CFG["metrics"] = [k for k, v in {
            "bp": bp_on, "hr": hr_on, "temp": temp_on, "sugar": sugar_on,
            "spo2": spo2_on, "rr": rr_on, "weight": weight_on, "waist": waist_on, "bmi": bmi_on
        }.items() if v]
        save_json(CONFIG_FILE, CFG)
        st.success("지표 설정 저장 완료")

    st.markdown("---")
    st.subheader("🔔 임계치 설정")
    thr = CFG.get("thresholds", DEFAULT_CONFIG["thresholds"])
    c1, c2 = st.columns(2)
    with c1:
        thr["bp_sys_hi"] = st.number_input("수축기 고혈압 ≥", value=int(thr["bp_sys_hi"]))
        thr["hr_lo"]     = st.number_input("심박 낮음 <",     value=int(thr["hr_lo"]))
        thr["temp_hi"]   = st.number_input("고열 ≥",          value=float(thr["temp_hi"]))
        thr["sugar_hi"]  = st.number_input("혈당 높음 ≥",      value=int(thr["sugar_hi"]))
        thr["sugar_lo"]  = st.number_input("저혈당 ≤",        value=int(thr["sugar_lo"]))
    with c2:
        thr["bp_dia_hi"] = st.number_input("이완기 고혈압 ≥", value=int(thr["bp_dia_hi"]))
        thr["hr_hi"]     = st.number_input("심박 높음 >",      value=int(thr["hr_hi"]))
        thr["sugar_very"]= st.number_input("혈당 위험 ≥",      value=int(thr["sugar_very"]))
        thr["spo2_lo"]   = st.number_input("SpO₂ 낮음 <",     value=int(thr["spo2_lo"]))
        thr["rr_lo"]     = st.number_input("호흡수 낮음 <",    value=int(thr["rr_lo"]))
        thr["rr_hi"]     = st.number_input("호흡수 높음 >",    value=int(thr["rr_hi"]))
    if st.button("저장(임계치)"):
        CFG["thresholds"] = thr
        save_json(CONFIG_FILE, CFG)
        st.success("임계치 저장 완료")

    st.markdown("---")
    st.subheader("👤 프로필 (BMI 계산)")
    height_cm = st.number_input("키(cm)", min_value=0.0, step=0.1, value=float(PROFILE.get("height_cm") or 0.0))
    if st.button("프로필 저장"):
        PROFILE["height_cm"] = height_cm if height_cm > 0 else None
        save_json(PROFILE_FILE, PROFILE)
        st.success("프로필 저장 완료")

# ---------------- 입력 폼 ----------------
st.markdown("### 📥 오늘의 지표 입력 (단위 포함)")
active_metrics = CFG.get("metrics", DEFAULT_CONFIG["metrics"]) or []

with st.form("entry_form"):
    c1, c2, c3 = st.columns(3)
    values: Dict[str, Any] = {}

    def render_metric(key: str, col):
        meta = METRIC_META[key]
        with col:
            if meta["type"] == "text":
                values[key] = st.text_input(meta["label"], placeholder=meta.get("placeholder", ""))
            elif meta["type"] == "int":
                values[key] = st.number_input(meta["label"], min_value=0, step=1)
            elif meta["type"] == "float":
                values[key] = st.number_input(meta["label"], min_value=0.0, step=meta.get("step", 0.1))

    cols_cycle = [c1, c2, c3]
    for idx, m in enumerate(active_metrics):
        render_metric(m, cols_cycle[idx % 3])

    memo = st.text_area("메모", placeholder="증상/변화/처치 간단 메모")

    # 날짜 / 시간 (시간은 한 박스 텍스트, 폭 축소)
    col_date, col_time = st.columns([1, 1])
    with col_date:
        record_date = st.date_input("기록 날짜", value=datetime.now().date())
    with col_time:
        default_time_str = datetime.now().strftime("%H:%M")

        # ⬇️ 래퍼 시작 (.time-narrow)
        st.markdown('<div class="time-narrow">', unsafe_allow_html=True)

        time_str = st.text_input(
            "기록 시간",
            value=default_time_str,
            placeholder="예: 09:30 / 0930",
            key="time_str"   # (임의의 키)
        )

        # ⬇️ 래퍼 끝
        st.markdown('</div>', unsafe_allow_html=True)

        # 문자열 → time 파싱
        try:
            if time_str.isdigit() and len(time_str) == 4:   # 0930 같은 형식
                t_obj = datetime.strptime(time_str, "%H%M").time()
            else:
                t_obj = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            t_obj = None
            st.caption("⚠ 시간 형식이 올바르지 않습니다. 예: 09:30 또는 0930")

    record_time = t_obj if t_obj else datetime.now().time()
    record_datetime = datetime.combine(record_date, record_time)

    use_sound = st.checkbox("임계치 초과 시 효과음 재생(브라우저 정책에 따라 자동재생 제한 가능)", value=False)
    submitted = st.form_submit_button("저장")

if submitted:
    entry = {"ts": record_datetime.isoformat(timespec="seconds")}
    for m in active_metrics:
        v = values.get(m)
        if (isinstance(v, str) and v.strip() == "") or v is None:
            continue
        entry[m] = v

    # BMI 자동 계산
    if PROFILE.get("height_cm") and ("weight" in entry) and ("bmi" in active_metrics) and ("bmi" not in entry):
        h_m = float(PROFILE["height_cm"]) / 100
        if h_m > 0:
            entry["bmi"] = round(float(entry["weight"]) / (h_m ** 2), 2)

    if memo.strip():
        entry["memo"] = memo.strip()

    DB.setdefault("entries", []).append(entry)
    DB["entries"].sort(key=lambda x: x["ts"], reverse=True)
    save_json(DATA_FILE, DB)

    flags = abnormal_flags(entry, CFG["thresholds"]) or None
    if flags:
        st.warning(f"경고: {flags}")
        st.toast(f"경고: {flags}")
        wav = make_beep_wav()
        if use_sound and wav:
            st.audio(wav)
    else:
        st.success("기록 저장 완료!")

st.markdown("---")

# ---------------- 조회 & 그래프 ----------------
st.markdown("### 🔎 기록 조회 & 그래프")
right_now = date.today()
start_default = right_now - timedelta(days=14)
col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
with col_a:
    start_d = st.date_input("시작일", value=start_default)
with col_b:
    end_d = st.date_input("종료일", value=right_now)
with col_c:
    plot_options = make_plot_options(active_metrics) or ["hr"]
    metric_for_plot = st.selectbox("그래프 지표", options=plot_options, index=0)
with col_d:
    kw = st.text_input("키워드(메모)")
    st.caption("↓ 슬라이더로 빠르게 기간 조절")
    use_slider = st.checkbox("최근 N일 보기", value=True)
    days_range = st.slider("N(일)", min_value=3, max_value=90, value=14, step=1, disabled=not use_slider)

if use_slider:
    start_dt = datetime.combine(right_now - timedelta(days=days_range - 1), datetime.min.time())
    end_dt   = datetime.combine(right_now, datetime.max.time())
else:
    start_dt = datetime.combine(start_d, datetime.min.time())
    end_dt   = datetime.combine(end_d, datetime.max.time())

# 표
rows: List[Dict[str, Any]] = []
for r in DB.get("entries", []):
    try:
        ts = datetime.fromisoformat(r["ts"])
    except Exception:
        continue
    if not (start_dt <= ts <= end_dt):
        continue
    if kw and kw not in r.get("memo", ""):
        continue
    row = {"날짜": ts.strftime("%Y-%m-%d %H:%M")}
    for m in METRIC_META:
        if m in r:
            row[METRIC_META[m]["label"]] = r[m]
    row["경고"] = abnormal_flags(r, CFG["thresholds"])
    if "memo" in r:
        row["메모"] = r["memo"]
    rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=360)
else:
    st.info("조회 기간에 해당하는 기록이 없습니다")

# 그래프
series_x, series_y = [], []

def _parse_bp(bp):
    try:
        s, d = str(bp).split("/")
        return int(s.strip()), int(d.strip())
    except Exception:
        return None, None

for r in DB.get("entries", []):
    try:
        ts = datetime.fromisoformat(r["ts"])
    except Exception:
        continue
    if not (start_dt <= ts <= end_dt):
        continue

    if metric_for_plot in ("bp_sys", "bp_dia"):
        if "bp" in r:
            s, d = _parse_bp(r["bp"])
            val = s if metric_for_plot == "bp_sys" else d
            if isinstance(val, (int, float)):
                series_x.append(ts); series_y.append(float(val))
    else:
        v = r.get(metric_for_plot)
        if isinstance(v, (int, float)):
            series_x.append(ts); series_y.append(float(v))

if series_x:
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(series_x, series_y, marker="o")
    title, unit = PLOT_META.get(metric_for_plot, (metric_for_plot, ""))
    ax.set_title(title); ax.set_ylabel(unit); ax.grid(True, alpha=0.3)
    thr = CFG["thresholds"]; ymin, ymax = ax.get_ylim()

    if metric_for_plot == "hr":
        ax.axhspan(thr["hr_hi"], ymax, alpha=0.08)
        ax.axhspan(ymin, thr["hr_lo"], alpha=0.08)
    elif metric_for_plot == "temp":
        ax.axhspan(thr["temp_hi"], ymax, alpha=0.08)
    elif metric_for_plot == "sugar":
        ax.axhspan(thr["sugar_very"], ymax, alpha=0.08)
        ax.axhspan(thr["sugar_hi"], thr["sugar_very"], alpha=0.08)
        ax.axhspan(ymin, thr["sugar_lo"], alpha=0.08)
    elif metric_for_plot == "spo2":
        ax.axhspan(ymin, thr["spo2_lo"], alpha=0.08)
    elif metric_for_plot == "rr":
        ax.axhspan(thr["rr_hi"], ymax, alpha=0.08)
        ax.axhspan(ymin, thr["rr_lo"], alpha=0.08)
    elif metric_for_plot == "bp_sys":
        ax.axhspan(thr["bp_sys_very"], ymax, alpha=0.08)
        ax.axhspan(thr["bp_sys_hi"], thr["bp_sys_very"], alpha=0.08)
    elif metric_for_plot == "bp_dia":
        ax.axhspan(thr["bp_dia_very"], ymax, alpha=0.08)
        ax.axhspan(thr["bp_dia_hi"], thr["bp_dia_very"], alpha=0.08)

    st.pyplot(fig)

st.markdown("---")

# ---------------- 기록 삭제 (선택/기간/전체) ----------------
st.markdown("### 🗑️ 기록 삭제")
tab_sel, tab_rng, tab_all = st.tabs(["선택 삭제", "기간 삭제", "전체 삭제"])

with tab_sel:
    _now = datetime.now()
    _start = _now - timedelta(days=30)
    _rows = []
    for r in DB.get("entries", []):
        try:
            ts = datetime.fromisoformat(r["ts"])
        except Exception:
            continue
        if ts < _start:
            continue
        row = {"id": r.get("id"), "기록시각": ts.strftime("%Y-%m-%d %H:%M")}
        for m in ["bp","hr","temp","sugar","spo2","rr","weight","bmi"]:
            if m in r: row[METRIC_META[m]["label"]] = r[m]
        row["메모"] = r.get("memo","")
        _rows.append(row)

    if _rows:
        df_edit = pd.DataFrame(_rows)
        df_edit.insert(0, "삭제", False)
        edited = st.data_editor(
            df_edit, use_container_width=True, height=420,
            column_config={"삭제": st.column_config.CheckboxColumn(),
                           "id": st.column_config.TextColumn("id", width="small")},
            hide_index=True,
        )
        ids_to_del = edited.loc[edited["삭제"]==True, "id"].dropna().tolist()
        if st.button("선택 항목 삭제", type="primary", disabled=(len(ids_to_del)==0)):
            before = len(DB["entries"])
            DB["entries"] = [e for e in DB["entries"] if e.get("id") not in ids_to_del]
            save_json(DATA_FILE, DB)
            st.success(f"{len(ids_to_del)}건 삭제 완료")
            st.rerun()
    else:
        st.info("최근 30일 내 표시할 기록이 없습니다.")

with tab_rng:
    c1, c2 = st.columns(2)
    with c1:
        del_start = st.date_input("삭제 시작일", value=(date.today()-timedelta(days=7)))
    with c2:
        del_end = st.date_input("삭제 종료일", value=date.today())
    del_start_dt = datetime.combine(del_start, datetime.min.time())
    del_end_dt   = datetime.combine(del_end,   datetime.max.time())

    cand = []
    for e in DB.get("entries", []):
        try:
            ts = datetime.fromisoformat(e["ts"])
        except Exception:
            continue
        if del_start_dt <= ts <= del_end_dt:
            cand.append(e)
    st.write(f"삭제 대상 미리보기: **{len(cand)}건**")

    confirm_rng = st.checkbox("정말 삭제하겠습니다(기간 삭제)")
    if st.button("기간 내 모두 삭제", type="primary", disabled=not confirm_rng):
        DB["entries"] = [e for e in DB["entries"] if e not in cand]
        save_json(DATA_FILE, DB)
        st.success(f"{len(cand)}건 삭제 완료")
        st.rerun()

with tab_all:
    st.error("⚠️ 주의: 전체 삭제는 되돌릴 수 없습니다.")
    confirm_text = st.text_input("확인 문구로 DELETE 를 입력하세요", value="")
    if st.button("모든 기록 삭제", type="primary", disabled=(confirm_text.strip()!="DELETE")):
        DB["entries"] = []
        save_json(DATA_FILE, DB)
        st.success("모든 기록을 삭제했습니다.")
        st.rerun()

# ---------------- PDF ----------------
st.markdown("### 🧾 리포트(PDF) — 주간/월간")
report_span = st.selectbox("기간", ["최근 7일", "최근 30일"], index=0)
span_days = 7 if report_span == "최근 7일" else 30
rep_start = date.today() - timedelta(days=span_days-1)
rep_end   = date.today()

if st.button("PDF 생성"):
    xs, hr_v, temp_v, sugar_v = [], [], [], []
    for r in DB.get("entries", []):
        try:
            ts = datetime.fromisoformat(r["ts"]).date()
        except Exception:
            continue
        if not (rep_start <= ts <= rep_end):
            continue
        xs.append(ts)
        hr_v.append(r.get("hr")); temp_v.append(r.get("temp")); sugar_v.append(r.get("sugar"))

    pdf_io = BytesIO()
    with PdfPages(pdf_io) as pdf:
        # 요약
        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis('off')
        lines = [ "바디로그 리포트",
                  f"기간: {rep_start.isoformat()} ~ {rep_end.isoformat()}","" ]
        def stat_line(name, arr):
            vals = [float(v) for v in arr if isinstance(v, (int, float))]
            return f"- {name}: 평균 {np.mean(vals):.1f}, 최솟값 {np.min(vals):.1f}, 최댓값 {np.max(vals):.1f}" if vals else f"- {name}: 데이터 없음"
        lines += [stat_line("심박수(bpm)", hr_v), stat_line("체온(°C)", temp_v), stat_line("혈당(mg/dL)", sugar_v)]
        plt.text(0.1, 0.9, "\n".join(lines), fontsize=14, va='top'); pdf.savefig(fig); plt.close(fig)

        def plot_series(dates, vals, title, unit, shade_cb=None):
            fig, ax = plt.subplots(figsize=(8.27, 4))
            dd, vv = [], []
            for d, v in zip(dates, vals):
                if isinstance(v, (int, float)): dd.append(d); vv.append(float(v))
            if dd:
                ax.plot(dd, vv, marker='o'); ax.set_title(title); ax.set_ylabel(unit); ax.grid(True, alpha=0.3)
                if shade_cb: shade_cb(ax)
            else:
                ax.text(0.5,0.5,'데이터 없음', ha='center', va='center'); ax.set_axis_off()
            pdf.savefig(fig); plt.close(fig)

        thr = CFG["thresholds"]
        plot_series(xs, hr_v,   "심박수(bpm)", "bpm", lambda ax: (ax.axhspan(thr["hr_hi"], ax.get_ylim()[1], alpha=0.08), ax.axhspan(ax.get_ylim()[0], thr["hr_lo"], alpha=0.08)))
        plot_series(xs, temp_v, "체온(°C)",    "°C",  lambda ax: ax.axhspan(thr["temp_hi"], ax.get_ylim()[1], alpha=0.08))
        plot_series(xs, sugar_v,"혈당(mg/dL)","mg/dL",lambda ax: (ax.axhspan(thr["sugar_very"], ax.get_ylim()[1], alpha=0.08), ax.axhspan(thr["sugar_hi"], thr["sugar_very"], alpha=0.08), ax.axhspan(ax.get_ylim()[0], thr["sugar_lo"], alpha=0.08)))
    pdf_io.seek(0)
    st.download_button("PDF 다운로드", data=pdf_io.getvalue(), file_name=f"bodylog_report_{rep_start.isoformat()}_{rep_end.isoformat()}.pdf", mime="application/pdf")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("단위 안내 — 혈압: mmHg | 호흡수: /min | 체온: °C | 심박수: bpm | SpO₂: % | 혈당: mg/dL | 기록 날짜: YYYY-MM-DD | 메모: 자유 기입")
