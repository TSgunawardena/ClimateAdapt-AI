"""ClimateAdapt AI local Streamlit prototype dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.agent import PlaceAnalysisOrchestrator  # noqa: E402
from climateadapt.place_analysis import PlaceAnalysisService  # noqa: E402

WORKFLOW_DIR = PROJECT_ROOT / "outputs" / "agent_workflow_v1"
WORKFLOW_STATE_PATH = WORKFLOW_DIR / "workflow_state.json"
AGENT_REPORT_PATH = WORKFLOW_DIR / "waimea_monaco_agent_report.json"
AGENT_REPORT_MARKDOWN = WORKFLOW_DIR / "waimea_monaco_agent_report.md"
REVIEW_PATH = WORKFLOW_DIR / "human_review_decision.json"

DEMO_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "demonstration_v2"
    / "waimea_monaco_demo_summary_v2.json"
)
TEST_METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "test_v2" / "tahunanui_test_metrics_v2.json"
)
VALIDATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation_v2"
    / "selected_threshold_v2.json"
)

DEMO_FIGURE = PROJECT_ROOT / "evidence" / "figures" / "waimea_monaco_demonstration_v2.png"
TEST_FIGURE = PROJECT_ROOT / "evidence" / "figures" / "tahunanui_test_evaluation_v2.png"
TEST_CURVES = PROJECT_ROOT / "evidence" / "figures" / "tahunanui_test_evaluation_curves_v2.png"
VALIDATION_FIGURE = PROJECT_ROOT / "evidence" / "figures" / "validation_threshold_selection_v2.png"
TRAINING_FIGURE = PROJECT_ROOT / "evidence" / "figures" / "unet_v2_training_curves.png"
FULL_REGION_FIGURE = (
    PROJECT_ROOT / "evidence" / "figures" / "nelson_full_region_inference_v2.png"
)


st.set_page_config(
    page_title="ClimateAdapt AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: rgba(18, 68, 89, 0.06);
        border: 1px solid rgba(18, 68, 89, 0.16);
        border-radius: 14px;
        padding: 16px;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, #073b4c 0%, #087f8c 55%, #22a699 100%);
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .92;}
    .guardrail {
        border-left: 5px solid #e09f3e;
        background: rgba(224, 159, 62, .10);
        padding: .9rem 1rem;
        border-radius: 8px;
    }
    .agent-card {
        border: 1px solid rgba(120,120,120,.22);
        padding: 1rem;
        border-radius: 12px;
        min-height: 145px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path, required: bool = False) -> dict:
    if not path.exists():
        if required:
            st.error(f"Required file is missing: `{path.relative_to(PROJECT_ROOT)}`")
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        st.error(f"Could not read `{path.name}`: {error}")
        return {}


def value_or(mapping: dict, key: str, default=0.0):
    value = mapping.get(key, default)
    return default if value is None else value


def format_number(value, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "Not available"


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: `{path.relative_to(PROJECT_ROOT)}`")


def download_button_for_file(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
        )


def status_label(status: str) -> str:
    mapping = {
        "human_review_required": "🟠 Human review required",
        "data_validated": "🟢 Data validated",
        "susceptibility_analysed": "🟢 Analysis completed",
        "failed": "🔴 Workflow failed",
    }
    return mapping.get(status, f"🔵 {status.replace('_', ' ').title()}")


workflow_state = read_json(WORKFLOW_STATE_PATH, required=True)
agent_report = read_json(AGENT_REPORT_PATH)
demo_summary = read_json(DEMO_SUMMARY_PATH)
test_metrics = read_json(TEST_METRICS_PATH)
validation_record = read_json(VALIDATION_PATH)
review_record = read_json(REVIEW_PATH)

facts = workflow_state.get("facts", {})
analysis = facts.get("flood_analysis", agent_report.get("flood_analysis", {}))
data_validation = facts.get("data_validation", agent_report.get("data_validation", {}))
recommendations = workflow_state.get(
    "recommendations", agent_report.get("planning_recommendations", [])
)
workflow_status = workflow_state.get("status", "not_run")


with st.sidebar:
    st.title("ClimateAdapt AI")
    st.caption("Coastal susceptibility planning prototype")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Place analysis",
            "Geospatial evidence",
            "Agent workflow",
            "Model evidence",
            "Human review",
        ],
    )
    st.divider()
    st.write("**Workflow status**")
    st.write(status_label(workflow_status))
    if workflow_state.get("run_id"):
        st.caption(f"Run ID: {workflow_state['run_id']}")
    st.divider()
    if st.button("Run / refresh three agents", use_container_width=True):
        try:
            from climateadapt.agent import ClimateAdaptOrchestrator

            with st.spinner("Running the three-agent workflow..."):
                result = ClimateAdaptOrchestrator(PROJECT_ROOT).run()
            if result.status == "failed":
                st.error("The workflow stopped because an agent reported an error.")
            else:
                st.success("Workflow refreshed. Human review is required.")
            st.rerun()
        except Exception as error:
            st.error(f"Workflow execution failed: {type(error).__name__}: {error}")


st.markdown(
    """
    <div class="hero">
      <h1>ClimateAdapt AI</h1>
      <p>Human-governed coastal susceptibility screening across the Nelson regional coverage</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "Overview":
    susceptible_area = value_or(
        analysis,
        "susceptible_area_square_kilometres",
        demo_summary.get("susceptible_area_square_kilometres", 0),
    )
    susceptible_percentage = value_or(
        analysis,
        "susceptible_percentage",
        demo_summary.get("susceptible_percentage", 0),
    )
    threshold = value_or(
        analysis,
        "operational_threshold",
        demo_summary.get("operational_threshold", 0),
    )
    uncertainty_percentage = value_or(
        analysis, "high_uncertainty_percentage", 0
    )

    columns = st.columns(4)
    columns[0].metric("Susceptible area", f"{format_number(susceptible_area, 3)} km²")
    columns[1].metric(
        "Susceptible proportion", f"{format_number(susceptible_percentage, 2)}%"
    )
    columns[2].metric("Frozen threshold", format_number(threshold, 2))
    columns[3].metric(
        "High-uncertainty cells", f"{format_number(uncertainty_percentage, 2)}%"
    )

    st.markdown("### Demonstration map")
    show_image(
        DEMO_FIGURE,
        "Label-blind Waimea Inlet / Monaco susceptibility demonstration",
    )

    st.markdown(
        """
        <div class="guardrail">
          <strong>Decision guardrail:</strong> This is a planning-screening prototype,
          not a validated flood extent, hydrodynamic simulation, regulatory product,
          or emergency-warning system. Final decisions require qualified human review.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Priority planning observations")
    if recommendations:
        for item in recommendations:
            st.markdown(
                f"**{item.get('priority', '•')}. {item.get('action', 'Review')}**  \n"
                f"{item.get('reason', '')}"
            )
    else:
        st.info("Run the three-agent workflow to generate planning observations.")


elif page == "Place analysis":
    st.subheader("Analyse a Nelson location")
    st.caption(
        "This page reads Script 19's precomputed regional rasters. It does not "
        "rerun the U-Net, retrain the model or change the frozen threshold."
    )

    service = PlaceAnalysisService(PROJECT_ROOT)
    missing_place_files = service.missing_files()
    if missing_place_files:
        st.error(
            "Place analysis is unavailable because required regional files are missing: "
            + ", ".join(
                f"`{path.relative_to(PROJECT_ROOT)}`" for path in missing_place_files
            )
        )
    else:
        left, right = st.columns([2, 1])
        with left:
            place_query = st.text_input(
                "Place or address",
                value="Atawhai",
                help=(
                    "Atawhai, Nelson CBD, Tāhunanui, Stoke and Monaco work from "
                    "the offline catalogue. Other places use online geocoding."
                ),
            )
        with right:
            radius_label = st.selectbox(
                "Analysis radius",
                ["500 metres", "1 kilometre", "2 kilometres"],
                index=1,
            )
        radius_lookup = {
            "500 metres": 500,
            "1 kilometre": 1000,
            "2 kilometres": 2000,
        }

        with st.expander("Manual coordinate fallback"):
            use_manual_coordinates = st.checkbox(
                "Use latitude and longitude instead of place-name search"
            )
            coordinate_columns = st.columns(2)
            manual_latitude = coordinate_columns[0].number_input(
                "Latitude", value=-41.2479, format="%.6f", disabled=not use_manual_coordinates
            )
            manual_longitude = coordinate_columns[1].number_input(
                "Longitude", value=173.3277, format="%.6f", disabled=not use_manual_coordinates
            )

        analyse_clicked = st.button(
            "Analyse selected place",
            type="primary",
            use_container_width=True,
        )
        if analyse_clicked:
            try:
                with st.spinner("Locating the place and reading regional predictions..."):
                    place = service.resolve_place(
                        place_query,
                        manual_latitude if use_manual_coordinates else None,
                        manual_longitude if use_manual_coordinates else None,
                    )
                    place_result = service.analyse(place, radius_lookup[radius_label])

                if place_result.get("status") != "analysis_ready":
                    st.session_state.pop("place_analysis_run", None)
                    st.error(
                        "Analysis unavailable: "
                        + place_result.get("reason", "No evaluated data coverage exists.")
                    )
                else:
                    with st.spinner("Running the three place-analysis agents..."):
                        place_state = PlaceAnalysisOrchestrator(PROJECT_ROOT).run(
                            place_result
                        )
                    st.session_state["place_analysis_run"] = {
                        "result": place_result,
                        "state": place_state.to_dict(),
                    }
                    if place_state.status == "failed":
                        st.error("The place-agent workflow reported an error.")
                    else:
                        st.success(
                            "Place analysis completed. The result is awaiting human review."
                        )
            except Exception as error:
                st.error(f"Place analysis failed: {type(error).__name__}: {error}")

        active_run = st.session_state.get("place_analysis_run")
        if active_run:
            place_result = active_run["result"]
            place_state = active_run["state"]
            place_facts = place_state.get("facts", {})
            place_analysis = place_facts.get("flood_analysis", {})
            place = place_result["place"]

            st.divider()
            st.markdown(f"### {place['display_name']}")
            st.caption(
                f"Coordinate source: {place['source']} · "
                f"Latitude {place['latitude']:.6f}, longitude {place['longitude']:.6f} · "
                f"Run ID: {place_result['run_id']}"
            )

            st.map(
                pd.DataFrame(
                    {"lat": [place["latitude"]], "lon": [place["longitude"]]}
                ),
                latitude="lat",
                longitude="lon",
                zoom=12,
                use_container_width=True,
            )

            metrics = st.columns(5)
            metrics[0].metric(
                "Evaluated land",
                f"{format_number(place_result['evaluated_land_area_square_kilometres'], 3)} km²",
            )
            metrics[1].metric(
                "Susceptible area",
                f"{format_number(place_result['susceptible_area_square_kilometres'], 3)} km²",
            )
            metrics[2].metric(
                "Susceptible land",
                f"{format_number(place_result['susceptible_percentage_of_evaluated_land'], 2)}%",
            )
            metrics[3].metric(
                "Mean probability", format_number(place_result["mean_probability"], 3)
            )
            metrics[4].metric(
                "Mean uncertainty", format_number(place_result["mean_uncertainty"], 3)
            )

            coverage = place_result["search_circle_coverage_percentage"]
            st.progress(min(max(coverage / 100.0, 0.0), 1.0))
            st.caption(
                f"Evaluated-land coverage within the circular search area: {coverage:.2f}%. "
                "Water and cells without valid model inputs are excluded."
            )

            local_map_path = PROJECT_ROOT / place_result["artifacts"]["local_map"]
            show_image(local_map_path, "Local place-based susceptibility screening")

            st.markdown("### Three-agent result")
            agent_columns = st.columns(3)
            place_agent_details = [
                (
                    "1. Data Validation Agent",
                    "Verified coverage, provenance, alignment and the frozen threshold.",
                    place_facts.get("data_validation", {}).get("status", "not available"),
                ),
                (
                    "2. Flood Analysis Agent",
                    "Calculated local susceptibility, probability and uncertainty statistics.",
                    "completed" if place_analysis else "not available",
                ),
                (
                    "3. Planning Support Agent",
                    "Created guarded observations and stopped for human review.",
                    place_state.get("status", "not available"),
                ),
            ]
            for column, (title, description, status) in zip(
                agent_columns, place_agent_details
            ):
                with column:
                    st.markdown(
                        f"<div class='agent-card'><strong>{title}</strong><br><br>"
                        f"{description}<br><br><em>Status: {status}</em></div>",
                        unsafe_allow_html=True,
                    )

            for warning in place_state.get("warnings", []):
                st.warning(warning)

            st.markdown("### Planning observations")
            for recommendation in place_state.get("recommendations", []):
                st.markdown(
                    f"**{recommendation.get('priority')}. "
                    f"{recommendation.get('action')}**  \n"
                    f"{recommendation.get('reason')}"
                )

            st.markdown(
                """
                <div class="guardrail">
                  <strong>Mandatory guardrail:</strong> This place result is a
                  screening indicator, not a validated flood extent, engineering
                  design, regulatory decision or emergency warning. A qualified
                  human must review the evidence before any action.
                </div>
                """,
                unsafe_allow_html=True,
            )

            artifacts = place_state.get("artifacts", {})
            st.markdown("### Download place evidence")
            download_columns = st.columns(3)
            with download_columns[0]:
                if artifacts.get("agent_report_markdown"):
                    download_button_for_file(
                        PROJECT_ROOT / artifacts["agent_report_markdown"],
                        "Download planning brief",
                        "text/markdown",
                    )
            with download_columns[1]:
                if artifacts.get("agent_report_json"):
                    download_button_for_file(
                        PROJECT_ROOT / artifacts["agent_report_json"],
                        "Download agent report",
                        "application/json",
                    )
            with download_columns[2]:
                if artifacts.get("analysis_json"):
                    download_button_for_file(
                        PROJECT_ROOT / artifacts["analysis_json"],
                        "Download local statistics",
                        "application/json",
                    )

            st.markdown("### Human review checkpoint")
            with st.form(f"place_review_{place_result['run_id']}"):
                review_columns = st.columns(2)
                place_reviewer_role = review_columns[0].selectbox(
                    "Reviewer role",
                    [
                        "Planning reviewer",
                        "GIS reviewer",
                        "Engineering reviewer",
                        "Academic assessor",
                    ],
                    key=f"place_role_{place_result['run_id']}",
                )
                place_review_reference = review_columns[1].text_input(
                    "Non-personal review reference",
                    placeholder="Example: PLACE-REVIEW-001",
                )
                place_decision = st.selectbox(
                    "Decision",
                    [
                        "Further investigation approved",
                        "Changes requested",
                        "Output rejected",
                    ],
                    key=f"place_decision_{place_result['run_id']}",
                )
                place_notes = st.text_area(
                    "Review notes",
                    placeholder="Record the evidence considered and required follow-up.",
                )
                place_acknowledgement = st.checkbox(
                    "I understand this is not a validated flood extent or emergency-warning product.",
                    key=f"place_ack_{place_result['run_id']}",
                )
                place_submitted = st.form_submit_button(
                    "Record place review", use_container_width=True
                )

            if place_submitted:
                if (
                    not place_review_reference.strip()
                    or not place_notes.strip()
                    or not place_acknowledgement
                ):
                    st.error(
                        "Provide a review reference, notes and the required acknowledgement."
                    )
                else:
                    decision_key = {
                        "Further investigation approved": "further_investigation_approved",
                        "Changes requested": "changes_requested",
                        "Output rejected": "rejected",
                    }[place_decision]
                    analysis_path = PROJECT_ROOT / artifacts["analysis_json"]
                    place_review_path = analysis_path.parent / "human_review_decision.json"
                    place_review_payload = {
                        "workflow_run_id": place_result["run_id"],
                        "area": place["display_name"],
                        "reviewer_role": place_reviewer_role,
                        "review_reference": place_review_reference.strip(),
                        "decision": decision_key,
                        "notes": place_notes.strip(),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "acknowledged_limitations": True,
                    }
                    place_review_path.write_text(
                        json.dumps(place_review_payload, indent=2), encoding="utf-8"
                    )
                    st.success("Place-analysis human-review decision recorded.")


elif page == "Geospatial evidence":
    st.subheader("Full Nelson regional inference")
    show_image(
        FULL_REGION_FIGURE,
        "Precomputed Experiment 2 regional probability, susceptibility and uncertainty",
    )

    st.subheader("Waimea Inlet / Monaco demonstration")
    st.caption(
        "The demonstration is label-blind. It visualises model susceptibility and "
        "an uncertainty indicator without making an accuracy claim."
    )
    show_image(DEMO_FIGURE, "Final weighted-overlap demonstration output")

    st.subheader("Generated geospatial artifacts")
    artifacts = workflow_state.get("artifacts", {})
    rows = []
    for name in ["probability", "binary", "uncertainty", "land_cover"]:
        relative_path = artifacts.get(name)
        if relative_path:
            path = PROJECT_ROOT / relative_path
            rows.append(
                {
                    "Artifact": name.replace("_", " ").title(),
                    "Path": relative_path,
                    "Available": path.exists(),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "The entropy layer is a probability-confidence indicator. It is not a "
        "complete estimate of epistemic or real-world flood uncertainty."
    )


elif page == "Agent workflow":
    st.subheader("Three-agent orchestration")
    columns = st.columns(3)
    agent_details = [
        (
            "1. Data Validation Agent",
            "Checks required files, provenance, EPSG:2193, raster alignment and the frozen threshold.",
            data_validation.get("status", "not available"),
        ),
        (
            "2. Flood Analysis Agent",
            "Calculates susceptibility, probability, uncertainty and land-cover statistics.",
            "completed" if analysis else "not available",
        ),
        (
            "3. Planning Support Agent",
            "Creates guarded planning observations and stops for human review.",
            workflow_status,
        ),
    ]
    for column, (title, description, status) in zip(columns, agent_details):
        with column:
            st.markdown(
                f"<div class='agent-card'><strong>{title}</strong><br><br>"
                f"{description}<br><br><em>Status: {status}</em></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Audit trail")
    audit_log = workflow_state.get("audit_log", [])
    if audit_log:
        audit_frame = pd.DataFrame(audit_log)
        st.dataframe(audit_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No audit events are available.")

    warnings = workflow_state.get("warnings", [])
    errors = workflow_state.get("errors", [])
    left, right = st.columns(2)
    left.metric("Warnings", len(warnings))
    right.metric("Errors", len(errors))
    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)

    with st.expander("Shared workflow state"):
        st.json(workflow_state)


elif page == "Model evidence":
    st.subheader("Experiment 2 evidence")
    tabs = st.tabs(["Training", "Validation", "Independent test", "Test curves"])
    with tabs[0]:
        show_image(TRAINING_FIGURE, "Experiment 2 training curves")
    with tabs[1]:
        show_image(VALIDATION_FIGURE, "Rocks Road validation threshold selection")
        if validation_record:
            left, right = st.columns(2)
            left.metric(
                "Operational threshold",
                format_number(
                    validation_record.get(
                        "operational_threshold",
                        validation_record.get("selected_threshold"),
                    ),
                    2,
                ),
            )
            right.metric(
                "Dice comparison threshold",
                format_number(validation_record.get("dice_optimal_threshold"), 2),
            )
    with tabs[2]:
        show_image(TEST_FIGURE, "Independent Tahunanui test evaluation")
        operational = test_metrics.get("operational_safety_threshold_metrics", {})
        threshold_free = test_metrics.get("threshold_free_metrics", {})
        columns = st.columns(4)
        columns[0].metric("Test Dice", format_number(operational.get("dice"), 3))
        columns[1].metric(
            "Test precision", format_number(operational.get("precision"), 3)
        )
        columns[2].metric(
            "Test recall", format_number(operational.get("recall_sensitivity"), 3)
        )
        columns[3].metric(
            "Average precision",
            format_number(threshold_free.get("average_precision_pr_auc"), 3),
        )
    with tabs[3]:
        show_image(TEST_CURVES, "ROC, precision–recall and calibration evidence")


elif page == "Human review":
    st.subheader("Human decision checkpoint")
    st.write(
        "The agents have completed screening and stopped. A human reviewer may "
        "record whether the result can proceed to further investigation."
    )

    if review_record:
        decision = review_record.get("decision", "not recorded")
        st.success(f"Current recorded decision: {decision.replace('_', ' ').title()}")
        st.caption(
            f"Recorded at {review_record.get('recorded_at_utc', 'unknown time')} "
            f"by role {review_record.get('reviewer_role', 'unspecified')}"
        )

    with st.form("human_review_form"):
        reviewer_role = st.selectbox(
            "Reviewer role",
            [
                "Planning reviewer",
                "GIS reviewer",
                "Engineering reviewer",
                "Academic assessor",
            ],
        )
        review_reference = st.text_input(
            "Non-personal review reference",
            placeholder="Example: REVIEW-001",
        )
        decision = st.selectbox(
            "Decision",
            [
                "Further investigation approved",
                "Changes requested",
                "Output rejected",
            ],
        )
        notes = st.text_area(
            "Review notes",
            placeholder="Record the evidence considered and any required follow-up.",
        )
        acknowledgement = st.checkbox(
            "I understand this is not a validated flood extent or emergency-warning product."
        )
        submitted = st.form_submit_button(
            "Record human review", use_container_width=True
        )

    if submitted:
        if not review_reference.strip() or not notes.strip() or not acknowledgement:
            st.error(
                "Provide a non-personal reference, review notes and the required acknowledgement."
            )
        else:
            decision_key = {
                "Further investigation approved": "further_investigation_approved",
                "Changes requested": "changes_requested",
                "Output rejected": "rejected",
            }[decision]
            payload = {
                "workflow_run_id": workflow_state.get("run_id"),
                "reviewer_role": reviewer_role,
                "review_reference": review_reference.strip(),
                "decision": decision_key,
                "notes": notes.strip(),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "acknowledged_limitations": True,
                "scope": (
                    "Decision applies only to whether the prototype output may "
                    "proceed to further investigation."
                ),
            }
            REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
            with REVIEW_PATH.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
            st.success("Human-review decision recorded.")
            st.rerun()

    st.subheader("Download evidence")
    columns = st.columns(3)
    with columns[0]:
        download_button_for_file(
            AGENT_REPORT_MARKDOWN, "Download planning brief", "text/markdown"
        )
    with columns[1]:
        download_button_for_file(
            AGENT_REPORT_PATH, "Download agent report", "application/json"
        )
    with columns[2]:
        download_button_for_file(
            WORKFLOW_STATE_PATH, "Download audit state", "application/json"
        )

    st.warning(
        "Approval means only that further investigation may proceed. It is not "
        "approval for construction, land-use change, emergency action or public warning."
    )
