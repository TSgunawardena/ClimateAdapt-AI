"""ClimateAdapt AI V3 local Streamlit prototype dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langgraph.types import Command  # noqa: E402
from climateadapt.agent.live_place_workflow_v1 import (  # noqa: E402
    build_live_place_graph,
    http_json,
    new_run_id,
    utc_now,
)
from climateadapt.place_analysis import PlaceAnalysisService  # noqa: E402

API_URL = os.getenv("CLIMATEADAPT_API_URL", "http://127.0.0.1:8000").rstrip("/")
OLLAMA_URL = os.getenv("CLIMATEADAPT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("CLIMATEADAPT_OLLAMA_MODEL", "climateadapt-planner:4b")

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = PROJECT_ROOT / "evidence" / "figures" / "v3"
FULL_SUMMARY = OUTPUTS / "full_region_inference_v3" / "nelson_logistic_full_region_summary_v3.json"
TEST_SUMMARY = OUTPUTS / "test_v3" / "v3_locked_test_summary.json"
TEST_COMPARISON = OUTPUTS / "test_v3" / "v3_locked_test_model_comparison.csv"
TEST_AREAS = OUTPUTS / "test_v3" / "v3_locked_test_area_metrics.csv"
BASELINE_VALIDATION = OUTPUTS / "baselines_v3" / "baseline_validation_comparison_v3.csv"
TRAINING_HISTORY = OUTPUTS / "training_v3" / "training_history_v3.csv"
LIVE_OUTPUT_DIR = OUTPUTS / "live_agent_workflow_v1"

FULL_FIGURE = FIGURES / "nelson_full_region_logistic_v3.png"
SPLIT_FIGURE = FIGURES / "v3_geographic_split.png"
TILE_FIGURE = FIGURES / "v3_tile_distribution.png"
BASELINE_FIGURE = FIGURES / "v3_baseline_validation.png"
TRAINING_FIGURE = FIGURES / "unet_v3_training_curves.png"
VALIDATION_FIGURE = FIGURES / "validation_threshold_selection_v3.png"
TEST_FIGURE = FIGURES / "v3_locked_test_evaluation.png"
TEST_CURVES = FIGURES / "v3_locked_test_curves.png"

st.set_page_config(page_title="ClimateAdapt AI V3", page_icon="🌊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(18,68,89,.06); border: 1px solid
    rgba(18,68,89,.16); border-radius: 14px; padding: 16px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
    background: linear-gradient(120deg,#073b4c 0%,#087f8c 55%,#22a699 100%);
    margin-bottom: 1rem;}
    .hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;} .hero p {margin:0; opacity:.92;}
    .guardrail {border-left: 5px solid #e09f3e; background: rgba(224,159,62,.10);
    padding: .9rem 1rem; border-radius: 8px;}
    .agent-card {border:1px solid rgba(120,120,120,.22); padding:1rem;
    border-radius:12px; min-height:145px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def first_value(mapping: dict[str, Any], keys: list[str], default: Any = 0.0) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "Not available"


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: `{path.relative_to(PROJECT_ROOT)}`")


def download(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(label, path.read_bytes(), path.name, mime, use_container_width=True)


def get_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    return value if isinstance(value, dict) else {"message": str(value)}


def api_state() -> tuple[dict[str, Any], dict[str, Any], str | None]:
    try:
        return (
            http_json(f"{API_URL}/health", timeout=5),
            http_json(f"{API_URL}/model", timeout=5),
            None,
        )
    except Exception as error:
        return {}, {}, f"{type(error).__name__}: {error}"


full_summary = read_json(FULL_SUMMARY)
test_summary = read_json(TEST_SUMMARY)
health, model, api_error = api_state()

with st.sidebar:
    st.title("ClimateAdapt AI")
    st.caption("V3 human-governed susceptibility prototype")
    page = st.radio(
        "Navigation",
        ["Overview", "Place analysis", "Geospatial evidence", "Agent workflow", "Model evidence", "Human review"],
    )
    st.divider()
    st.write("**Live services**")
    if api_error:
        st.error("V3 prediction API unavailable")
        st.caption(api_error)
    else:
        st.success("V3 prediction API connected")
        st.caption(f"Model: {model.get('primary_model', 'unknown')}")
        st.caption(f"Threshold: {number(model.get('operational_threshold'), 4)}")
    st.caption(f"Planning LLM: {OLLAMA_MODEL}")

st.markdown(
    """<div class="hero"><h1>ClimateAdapt AI V3</h1>
    <p>Human-governed coastal susceptibility screening across the Nelson regional coverage</p></div>""",
    unsafe_allow_html=True,
)

if page == "Overview":
    evaluated_area = first_value(full_summary, ["evaluated_area_square_kilometres", "evaluated_land_area_square_kilometres"])
    susceptible_area = first_value(full_summary, ["susceptible_area_square_kilometres", "susceptible_area_km2"])
    susceptible_percentage = first_value(full_summary, ["susceptible_percentage", "susceptible_percentage_of_evaluated_land"])
    columns = st.columns(4)
    columns[0].metric("Evaluated land", f"{number(evaluated_area)} km²")
    columns[1].metric("Flagged area", f"{number(susceptible_area)} km²")
    columns[2].metric("Flagged proportion", f"{number(susceptible_percentage, 2)}%")
    columns[3].metric("Frozen threshold", number(model.get("operational_threshold", .004), 4))
    st.info("The deployed primary model is V3 logistic regression. Its susceptibility score is an uncalibrated screening score, not a flood probability.")
    show_image(FULL_FIGURE, "V3 full-region logistic susceptibility screening")
    st.markdown(
        """<div class="guardrail"><strong>Decision boundary:</strong> This prototype
        identifies locations for further review. It is not a validated flood extent,
        hydrodynamic simulation, engineering design, regulatory decision or emergency-warning service.</div>""",
        unsafe_allow_html=True,
    )

elif page == "Place analysis":
    st.subheader("Understand susceptibility around a Nelson place")
    st.caption(
        "Search for a place, choose an area around it, and receive a mapped V3 "
        "screening result. The result shows flagged land area—not just one raster cell."
    )
    service = PlaceAnalysisService(PROJECT_ROOT)
    missing_inputs = service.missing_files()
    if api_error:
        st.error("Start Script 30 before running the three-agent workflow.")
    if missing_inputs:
        st.error("The V3 area-analysis inputs are incomplete.")
        for missing in missing_inputs:
            st.code(str(missing.relative_to(PROJECT_ROOT)))
    else:
        first, second, third = st.columns([2, 1, 1])
        place_query = first.text_input("Place or address", value="Rocks Road")
        radius_label = second.selectbox(
            "Area around place",
            ["500 metres", "1 kilometre", "2 kilometres"],
            index=1,
        )
        reviewer = third.text_input("Reviewer label", value="Prototype reviewer")
        radius_lookup = {
            "500 metres": 500,
            "1 kilometre": 1000,
            "2 kilometres": 2000,
        }
        radius_metres = radius_lookup[radius_label]

        with st.expander("Manual coordinate option"):
            use_manual = st.checkbox("Use latitude and longitude")
            c1, c2 = st.columns(2)
            latitude = c1.number_input(
                "Latitude", value=-41.2750, format="%.6f", disabled=not use_manual
            )
            longitude = c2.number_input(
                "Longitude", value=173.2690, format="%.6f", disabled=not use_manual
            )

        if st.button(
            "Analyse area with the three agents",
            type="primary",
            use_container_width=True,
            disabled=bool(api_error),
        ):
            try:
                with st.spinner("Resolving the place and analysing the surrounding land..."):
                    place = service.resolve_place(
                        place_query,
                        latitude if use_manual else None,
                        longitude if use_manual else None,
                    )
                    area_result = service.analyse(place, radius_metres)
                if area_result.get("status") != "analysis_ready":
                    raise RuntimeError(
                        area_result.get("reason", "The area could not be analysed.")
                    )

                run_id = area_result["run_id"]
                graph = build_live_place_graph()
                config = {"configurable": {"thread_id": run_id}}
                initial = {
                    "project_root": str(PROJECT_ROOT),
                    "run_id": run_id,
                    "started_at_utc": utc_now(),
                    "location_name": place["display_name"],
                    "location_request": {
                        "location_name": place["display_name"],
                        "latitude": float(place["latitude"]),
                        "longitude": float(place["longitude"]),
                    },
                    "area_result": area_result,
                    "api_url": API_URL,
                    "ollama_url": OLLAMA_URL,
                    "ollama_model": OLLAMA_MODEL,
                    "status": "created",
                    "current_agent": "langgraph_orchestrator",
                    "warnings": [],
                    "errors": [],
                    "audit_log": [],
                    "artifacts": {},
                    "revision_count": 0,
                }
                with st.spinner("Running validation, analysis and planning agents..."):
                    result = graph.invoke(initial, config=config)
                st.session_state["v3_live_run"] = {
                    "graph": graph,
                    "config": config,
                    "result": result,
                    "place": place,
                    "area_result": area_result,
                    "reviewer": reviewer,
                }
                if result.get("errors") or result.get("status") == "failed":
                    st.error("The workflow stopped safely because validation failed.")
                else:
                    st.success("Area analysis reached the mandatory human-review checkpoint.")
            except Exception as error:
                st.session_state.pop("v3_live_run", None)
                st.error(f"V3 area analysis failed: {type(error).__name__}: {error}")

        active = st.session_state.get("v3_live_run")
        if active:
            result = active["result"]
            place = active["place"]
            area_result = active["area_result"]
            analysis = result.get("analysis", {})
            planning = result.get("planning_report", {})
            flagged = int(area_result["susceptible_cells"]) > 0

            st.divider()
            st.markdown(f"### {place['display_name']}")
            st.caption(
                f"{area_result['radius_metres']:,} m radius · "
                f"{place.get('source', 'resolved location')} · "
                f"Run ID: {result.get('run_id', area_result['run_id'])}"
            )
            st.map(
                pd.DataFrame(
                    {
                        "lat": [float(place["latitude"])],
                        "lon": [float(place["longitude"])],
                    }
                ),
                latitude="lat",
                longitude="lon",
                zoom=12,
                use_container_width=True,
            )

            if flagged:
                st.warning("Further investigation is recommended for this area.")
            else:
                st.info(
                    "No evaluated land cells crossed the frozen screening threshold "
                    "in this search area. This does not prove that flood risk is absent."
                )
            st.write(area_result["plain_language_summary"])

            metrics = st.columns(5)
            metrics[0].metric(
                "Land evaluated",
                f"{number(area_result['evaluated_land_area_square_kilometres'])} km²",
            )
            metrics[1].metric(
                "Flagged land",
                f"{number(area_result['susceptible_area_square_kilometres'])} km²",
            )
            metrics[2].metric(
                "Flagged proportion",
                f"{number(area_result['susceptible_percentage_of_evaluated_land'], 1)}%",
            )
            metrics[3].metric(
                "Land coverage",
                f"{number(area_result['evaluated_land_coverage_percentage'], 1)}%",
            )
            metrics[4].metric(
                "Frozen threshold",
                number(area_result["operational_threshold"], 4),
            )
            st.warning(
                "The V3 score is an uncalibrated susceptibility-screening score, "
                "not a percentage chance of flooding. The threshold only identifies "
                "land for further review."
            )

            local_map = PROJECT_ROOT / area_result["artifacts"]["local_map"]
            show_image(
                local_map,
                "Local V3 evidence: terrain, susceptibility score, flagged land and coast distance",
            )

            with st.expander("Technical score and terrain details"):
                detail_columns = st.columns(3)
                detail_columns[0].metric(
                    "Mean score",
                    number(area_result["mean_susceptibility_score"], 6),
                )
                detail_columns[1].metric(
                    "Median score",
                    number(area_result["median_susceptibility_score"], 6),
                )
                detail_columns[2].metric(
                    "Maximum score",
                    number(area_result["maximum_susceptibility_score"], 6),
                )
                terrain_columns = st.columns(3)
                terrain_columns[0].metric(
                    "Median elevation",
                    f"{number(area_result['median_elevation_metres'], 1)} m",
                )
                terrain_columns[1].metric(
                    "Median slope",
                    f"{number(area_result['median_slope_degrees'], 1)}°",
                )
                terrain_columns[2].metric(
                    "Median coast distance",
                    f"{number(area_result['median_distance_to_coast_metres'], 0)} m",
                )
                st.json(
                    {
                        "score_quantiles": area_result["susceptibility_score_quantiles"],
                        "risk_band_percentages": area_result["risk_band_percentages"],
                        "leading_land_cover_codes": area_result[
                            "leading_susceptible_land_cover_codes"
                        ],
                    }
                )

            st.markdown("### Three-agent result")
            cards = [
                (
                    "1. Data Validation Agent",
                    "Verified the V3 model, frozen threshold, coverage and provenance.",
                    result.get("validation_result", {}).get("status", "pending"),
                ),
                (
                    "2. Flood Analysis Agent",
                    "Interpreted flagged area, score distribution and terrain context.",
                    "completed" if analysis else "pending",
                ),
                (
                    "3. Planning Support Agent",
                    "Produced a guarded planning brief and stopped for human review.",
                    planning.get("status", "pending"),
                ),
            ]
            for column, (title, description, status) in zip(st.columns(3), cards):
                column.markdown(
                    f"<div class='agent-card'><strong>{title}</strong><br><br>"
                    f"{description}<br><br><em>Status: {status}</em></div>",
                    unsafe_allow_html=True,
                )

            if planning:
                st.markdown(f"### {planning.get('headline', 'Planning brief')}")
                st.write(planning.get("summary", ""))
                with st.expander("Evidence, actions and limitations"):
                    for heading, key in [
                        ("Evidence", "evidence"),
                        ("Recommended actions", "recommended_actions"),
                        ("Limitations", "limitations"),
                    ]:
                        st.markdown(f"**{heading}**")
                        for item in planning.get(key, []):
                            st.markdown(f"- {item}")

            for warning in result.get("warnings", []):
                st.warning(warning)
            for error in result.get("errors", []):
                st.error(error)

            checkpoint = get_interrupt(result)
            if checkpoint:
                st.markdown("### Human review checkpoint")
                st.info(checkpoint.get("message", "Human review is required."))
                with st.form(f"v3_review_{result.get('run_id', 'active')}"):
                    action = st.selectbox("Decision", ["approve", "revise", "reject"])
                    notes = st.text_area("Reviewer notes")
                    acknowledged = st.checkbox(
                        "I understand this is screening evidence, not a flood "
                        "probability, validated extent or emergency-warning product."
                    )
                    submitted = st.form_submit_button(
                        "Submit human decision", use_container_width=True
                    )
                if submitted:
                    if not notes.strip() or not acknowledged:
                        st.error("Provide review notes and acknowledge the limitations.")
                    else:
                        try:
                            with st.spinner("Recording the decision..."):
                                resumed = active["graph"].invoke(
                                    Command(
                                        resume={
                                            "action": action,
                                            "reviewer": active["reviewer"],
                                            "notes": notes.strip(),
                                        }
                                    ),
                                    config=active["config"],
                                )
                            active["result"] = resumed
                            st.session_state["v3_live_run"] = active
                            st.rerun()
                        except Exception as error:
                            st.error(
                                f"Could not record the review: "
                                f"{type(error).__name__}: {error}"
                            )
            else:
                if result.get("status") == "failed" or result.get("errors"):
                    st.error("Workflow stopped safely with status: failed")
                else:
                    st.success(
                        f"Workflow completed with status: "
                        f"{result.get('status', 'completed')}"
                    )
                artifacts = result.get("artifacts", {})
                downloads = [
                    (
                        "planning_report_markdown",
                        "Download planning brief",
                        "text/markdown",
                    ),
                    (
                        "workflow_state",
                        "Download audit state",
                        "application/json",
                    ),
                ]
                for column, (key, label, mime) in zip(st.columns(2), downloads):
                    path = artifacts.get(key)
                    if path:
                        with column:
                            download(PROJECT_ROOT / path, label, mime)

            analysis_json = PROJECT_ROOT / area_result["artifacts"]["analysis_json"]
            download(
                analysis_json,
                "Download complete area-analysis JSON",
                "application/json",
            )

elif page == "Geospatial evidence":
    st.subheader("V3 full Nelson regional screening")
    show_image(FULL_FIGURE, "Primary V3 logistic full-region susceptibility output")
    st.caption("The score raster was generated label-blind with frozen model parameters. Scores are not calibrated probabilities.")
    tabs = st.tabs(["Geographic split", "Tile distribution", "Locked test"])
    with tabs[0]: show_image(SPLIT_FIGURE, "Leakage-controlled V3 geographic split")
    with tabs[1]: show_image(TILE_FIGURE, "V3 train, validation and locked-test tiles")
    with tabs[2]: show_image(TEST_FIGURE, "Final locked geographic test")

elif page == "Agent workflow":
    st.subheader("V3 LangGraph three-agent workflow")
    cards = [
        ("1. Data Validation Agent", "Validates API health, CRS, coverage, model identity, provenance and threshold."),
        ("2. Flood Analysis Agent", "Interprets the uncalibrated score, threshold, terrain and uncertainty."),
        ("3. Planning Support Agent", "Calls climateadapt-planner:4b, validates JSON and requires human review."),
    ]
    for column, (title, description) in zip(st.columns(3), cards):
        column.markdown(f"<div class='agent-card'><strong>{title}</strong><br><br>{description}</div>", unsafe_allow_html=True)
    st.markdown("### Implemented stopping condition")
    st.write("Automation stops after producing a guarded draft. It cannot release the brief or perform a Gitea action without a recorded human decision.")
    active = st.session_state.get("v3_live_run")
    if active:
        audit = active["result"].get("audit_log", [])
        if audit: st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)
        with st.expander("Current workflow state"):
            st.json({k: v for k, v in active["result"].items() if k != "__interrupt__"})
    else:
        st.info("Run a location from Place analysis to populate the live audit trail.")

elif page == "Model evidence":
    st.subheader("V3 model selection and locked-test evidence")
    st.info("The U-Net beat logistic regression on validation AP (0.7209 vs 0.6619), but frozen logistic regression generalised substantially better on the locked geographic test (AP 0.9509 vs 0.6680). Logistic regression is therefore the deployed V3 model.")
    tabs = st.tabs(["Baseline validation", "U-Net training", "Thresholds", "Locked test", "Metrics"])
    with tabs[0]:
        show_image(BASELINE_FIGURE, "V3 baseline validation")
        frame = read_csv(BASELINE_VALIDATION)
        if not frame.empty: st.dataframe(frame, use_container_width=True, hide_index=True)
    with tabs[1]:
        show_image(TRAINING_FIGURE, "V3 GroupNorm Compact U-Net training")
        frame = read_csv(TRAINING_HISTORY)
        if not frame.empty: st.dataframe(frame, use_container_width=True, hide_index=True)
    with tabs[2]:
        show_image(VALIDATION_FIGURE, "V3 U-Net validation threshold selection")
        st.caption("U-Net operational threshold: 0.20. Deployed logistic threshold: 0.004. They belong to different models and are not interchangeable.")
    with tabs[3]:
        show_image(TEST_FIGURE, "V3 locked geographic test evaluation")
        show_image(TEST_CURVES, "V3 locked-test curves")
    with tabs[4]:
        pooled, areas = read_csv(TEST_COMPARISON), read_csv(TEST_AREAS)
        if not pooled.empty: st.dataframe(pooled, use_container_width=True, hide_index=True)
        if not areas.empty: st.dataframe(areas, use_container_width=True, hide_index=True)
        with st.expander("Locked-test summary JSON"): st.json(test_summary)

elif page == "Human review":
    st.subheader("Human decision records")
    st.write("Approval only permits further investigation. It does not approve engineering, regulation, funding, land-use change, public warning or emergency action.")
    files = sorted(LIVE_OUTPUT_DIR.glob("workflow_*.json"), key=lambda path: path.stat().st_mtime, reverse=True) if LIVE_OUTPUT_DIR.exists() else []
    if not files:
        st.info("No completed live V3 workflow record is available yet.")
    else:
        rows = []
        for path in files[:20]:
            record, decision = read_json(path), read_json(path).get("human_decision", {})
            rows.append({"Run ID": record.get("run_id"), "Location": record.get("location_name"), "Status": record.get("status"), "Decision": decision.get("action"), "Reviewer": decision.get("reviewer"), "Recorded UTC": decision.get("reviewed_at_utc"), "File": str(path.relative_to(PROJECT_ROOT))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Inspect record", [row["File"] for row in rows])
        selected_path = PROJECT_ROOT / selected
        with st.expander("Complete workflow evidence"): st.json(read_json(selected_path))
        download(selected_path, "Download selected audit record", "application/json")

