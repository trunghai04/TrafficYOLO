import os
from pathlib import Path
import tempfile
import cv2
import pandas as pd
import streamlit as st
import numpy as np

from traffic_yolo.config import load_config, get_default_device
from traffic_yolo.pipeline import TrafficMonitorPipeline
from traffic_yolo.evidence import create_combined_evidence, enhance_license_plate, detect_license_plate_crop
from traffic_yolo.traffic_light import (
    auto_detect_signals,
    classify_approach_signal,
    classify_right_pole_assembly,
)


st.set_page_config(
    page_title="TrafficYOLO - Giám Sát Giao Thông & Xử Phạt Vượt Đèn Đỏ",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚦 TrafficYOLO - Hệ Thống Giám Sát Giao Thông Thông Minh")
st.markdown(
    "Ứng dụng phân tích video giao thông tự động: Đếm lưu lượng phương tiện, "
    "giám sát tín hiệu đèn, bắt lỗi vượt vạch khi đèn đỏ và trích xuất bằng chứng biển số xe."
)

# Load configuration template
base_config = load_config()

# Initialize session state for line and ROIs
if "line_x1" not in st.session_state:
    st.session_state["line_x1"] = int(base_config["counting_line"]["start"][0])
    st.session_state["line_y1"] = int(base_config["counting_line"]["start"][1])
    st.session_state["line_x2"] = int(base_config["counting_line"]["end"][0])
    st.session_state["line_y2"] = int(base_config["counting_line"]["end"][1])

    st.session_state["roi1_x1"] = int(base_config["traffic_signals"]["straight_light"]["roi"][0])
    st.session_state["roi1_y1"] = int(base_config["traffic_signals"]["straight_light"]["roi"][1])
    st.session_state["roi1_x2"] = int(base_config["traffic_signals"]["straight_light"]["roi"][2])
    st.session_state["roi1_y2"] = int(base_config["traffic_signals"]["straight_light"]["roi"][3])

    st.session_state["roi2_x1"] = int(base_config["traffic_signals"]["right_turn_light"]["roi"][0])
    st.session_state["roi2_y1"] = int(base_config["traffic_signals"]["right_turn_light"]["roi"][1])
    st.session_state["roi2_x2"] = int(base_config["traffic_signals"]["right_turn_light"]["roi"][2])
    st.session_state["roi2_y2"] = int(base_config["traffic_signals"]["right_turn_light"]["roi"][3])

# =============================
# SIDEBAR: CẤU HÌNH THAM SỐ
# =============================
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")

    device_option = st.selectbox(
        "Thiết bị tính toán (Device)",
        options=["auto", "cuda", "cpu"],
        index=0,
        help="'auto': Tự nhận diện GPU nếu có, ngược lại dùng CPU",
    )

    conf_threshold = st.slider(
        "Ngưỡng tin cậy nhận diện (Confidence)",
        min_value=0.10,
        max_value=0.90,
        value=float(base_config["inference"]["confidence"]),
        step=0.05,
    )

    st.subheader("📹 Cấu hình Góc Camera")
    cam_preset = st.selectbox(
        "Mẫu góc quay (Camera Preset)",
        options=[
            "🎯 Tự Động Nhận Diện (Khuyên Dùng)",
            "📐 Góc Rộng (12h, 18h - Đèn rẽ phải mép phải x=1550)",
            "📐 Góc Chuẩn (16h15, 16h30 - Đèn rẽ phải x=1400)",
            "⚙️ Tùy chỉnh thủ công",
        ],
        index=0,
        help="Hệ thống có thể tự nhận diện góc quay hoặc bạn chọn cấu hình có sẵn cho từng video.",
    )

    if cam_preset == "📐 Góc Rộng (12h, 18h - Đèn rẽ phải mép phải x=1550)":
        st.session_state["roi1_x1"], st.session_state["roi1_y1"] = 1525, 345
        st.session_state["roi1_x2"], st.session_state["roi1_y2"] = 1585, 530
        st.session_state["roi2_x1"], st.session_state["roi2_y1"] = 1525, 345
        st.session_state["roi2_x2"], st.session_state["roi2_y2"] = 1585, 530
        st.session_state["line_x1"], st.session_state["line_y1"] = 0, 590
        st.session_state["line_x2"], st.session_state["line_y2"] = 1550, 650
    elif cam_preset == "📐 Góc Chuẩn (16h15, 16h30 - Đèn rẽ phải x=1400)":
        st.session_state["roi1_x1"], st.session_state["roi1_y1"] = 545, 15
        st.session_state["roi1_x2"], st.session_state["roi1_y2"] = 610, 85
        st.session_state["roi2_x1"], st.session_state["roi2_y1"] = 1375, 280
        st.session_state["roi2_x2"], st.session_state["roi2_y2"] = 1445, 415
        st.session_state["line_x1"], st.session_state["line_y1"] = 0, 520
        st.session_state["line_x2"], st.session_state["line_y2"] = 1500, 750

    st.subheader("🚦 Kiểu Hệ Thống Đèn Tín Hiệu")
    signal_system_mode = st.radio(
        "Loại cột đèn tại nút giao:",
        options=[
            "2 Cụm Đèn Riêng (Đi Thẳng + Rẽ Phải Riêng)",
            "Chỉ 1 Cụm Đèn Chung (Không có đèn rẽ phải)",
        ],
        index=0,
        help="Chọn 'Chỉ 1 Cụm Đèn Chung' nếu nút giao chỉ có 1 đèn duy nhất điều khiển toàn bộ phương tiện.",
    )
    is_single_light_ui = (signal_system_mode == "Chỉ 1 Cụm Đèn Chung (Không có đèn rẽ phải)")

    allow_right_on_red_ui = False
    if is_single_light_ui:
        allow_right_on_red_ui = st.checkbox(
            "Cho phép rẽ phải khi đèn đỏ (Biển phụ cho phép rẽ phải)",
            value=False,
            help="Tích chọn nếu nút giao có biển 'Đèn đỏ được phép rẽ phải'. Khi đó xe rẽ phải sẽ không bị tính là vi phạm.",
        )

    st.subheader("📏 Cấu hình Vạch Dừng Ảo")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        line_x1 = st.number_input("Start X", value=int(st.session_state["line_x1"]), step=10)
        line_y1 = st.number_input("Start Y", value=int(st.session_state["line_y1"]), step=10)
    with col_l2:
        line_x2 = st.number_input("End X", value=int(st.session_state["line_x2"]), step=10)
        line_y2 = st.number_input("End Y", value=int(st.session_state["line_y2"]), step=10)
    st.session_state["line_x1"], st.session_state["line_y1"] = line_x1, line_y1
    st.session_state["line_x2"], st.session_state["line_y2"] = line_x2, line_y2

    st.subheader("🚶‍♂️ Bộ Lọc Người Đi Bộ")
    night_rider_boost = st.checkbox(
        "Nhận diện người lái xe ban đêm (Night Rider Boost)",
        value=False,
        help="Mặc định TẮT để không bao giờ nhận diện nhầm người đi bộ qua đường là xe máy. Chỉ bật khi phân tích video ban đêm thiếu sáng bị mất bóng xe máy.",
    )

    if is_single_light_ui:
        st.subheader("💡 Đèn Tín Hiệu Chung (Traffic Light ROI)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            roi1_x1 = st.number_input("Light X1", value=int(st.session_state["roi1_x1"]), step=5)
            roi1_y1 = st.number_input("Light Y1", value=int(st.session_state["roi1_y1"]), step=5)
        with col_r2:
            roi1_x2 = st.number_input("Light X2", value=int(st.session_state["roi1_x2"]), step=5)
            roi1_y2 = st.number_input("Light Y2", value=int(st.session_state["roi1_y2"]), step=5)
        roi2_x1, roi2_y1, roi2_x2, roi2_y2 = roi1_x1, roi1_y1, roi1_x2, roi1_y2
        st.session_state["roi1_x1"], st.session_state["roi1_y1"] = roi1_x1, roi1_y1
        st.session_state["roi1_x2"], st.session_state["roi1_y2"] = roi1_x2, roi1_y2
        st.session_state["roi2_x1"], st.session_state["roi2_y1"] = roi2_x1, roi2_y1
        st.session_state["roi2_x2"], st.session_state["roi2_y2"] = roi2_x2, roi2_y2
    else:
        st.subheader("💡 Đèn 1: Đi Thẳng (Straight ROI)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            roi1_x1 = st.number_input("Straight X1", value=int(st.session_state["roi1_x1"]), step=5)
            roi1_y1 = st.number_input("Straight Y1", value=int(st.session_state["roi1_y1"]), step=5)
        with col_r2:
            roi1_x2 = st.number_input("Straight X2", value=int(st.session_state["roi1_x2"]), step=5)
            roi1_y2 = st.number_input("Straight Y2", value=int(st.session_state["roi1_y2"]), step=5)

        st.subheader("👉 Đèn 2: Rẽ Phải (Right Turn ROI)")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            roi2_x1 = st.number_input("Right Turn X1", value=int(st.session_state["roi2_x1"]), step=5)
            roi2_y1 = st.number_input("Right Turn Y1", value=int(st.session_state["roi2_y1"]), step=5)
        with col_t2:
            roi2_x2 = st.number_input("Right Turn X2", value=int(st.session_state["roi2_x2"]), step=5)
            roi2_y2 = st.number_input("Right Turn Y2", value=int(st.session_state["roi2_y2"]), step=5)

        st.session_state["roi1_x1"], st.session_state["roi1_y1"] = roi1_x1, roi1_y1
        st.session_state["roi1_x2"], st.session_state["roi1_y2"] = roi1_x2, roi1_y2
        st.session_state["roi2_x1"], st.session_state["roi2_y1"] = roi2_x1, roi2_y1
        st.session_state["roi2_x2"], st.session_state["roi2_y2"] = roi2_x2, roi2_y2

    save_video_out = st.checkbox("Xuất video kết quả (Annotated Video)", value=True)


# =============================
# CHỌN NGUỒN VIDEO
# =============================
st.subheader("📁 Chọn nguồn Video đầu vào")
input_mode = st.radio(
    "Phương thức cung cấp video:",
    options=["Nhập đường dẫn trên máy (Local Path)", "Tải lên file video (Upload)"],
    horizontal=True,
)

video_source_path = None
uploaded_temp_file = None

if input_mode == "Nhập đường dẫn trên máy (Local Path)":
    default_sample = "videos/3.mp4" if Path("videos/3.mp4").exists() else ""
    local_path_input = st.text_input(
        "Nhập đường dẫn file video (Ví dụ: C:/Videos/traffic.mp4 hoặc videos/3.mp4):",
        value=default_sample,
        placeholder="C:/path/to/traffic_video.mp4",
    )
    if local_path_input and Path(local_path_input).exists():
        video_source_path = Path(local_path_input)
        st.success(f"Đã tìm thấy video: `{video_source_path.resolve()}`")
    elif local_path_input:
        st.error(f"Không tìm thấy file: `{local_path_input}`")

else:
    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn file video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded_file is not None:
        upload_dir = Path("outputs/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        cached_path = upload_dir / uploaded_file.name

        # Save uploaded file once cleanly
        if not cached_path.exists() or cached_path.stat().st_size == 0:
            with open(cached_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        video_source_path = cached_path
        st.success(f"Tải lên video thành công: `{uploaded_file.name}` ({video_source_path.stat().st_size / (1024*1024):.1f} MB)")

# =============================
# NÚT BẤM BẮT ĐẦU CHÍNH (ALWAYS VISIBLE)
# =============================
st.write("")
col_act1, col_act2 = st.columns([1, 1])
with col_act1:
    start_btn_top = st.button(
        "🚀 BẮT ĐẦU PHÂN TÍCH GIAO THÔNG",
        type="primary",
        disabled=video_source_path is None,
        key="btn_start_top",
        use_container_width=True,
    )
with col_act2:
    if video_source_path:
        st.info(f"📹 Video đã sẵn sàng: `{video_source_path.name}`. Bấm nút màu đỏ bên cạnh để chạy ngay.")

# =============================
# XEM TRƯỚC VẠCH KẺ & ĐÈN ROI
# =============================
if video_source_path and video_source_path.exists():
    with st.expander("👁️ Xem trước khung hình & Căn chỉnh Vạch Dừng / Đèn ROI", expanded=True):
        cap = cv2.VideoCapture(str(video_source_path))
        ret, preview_frame = cap.read()
        cap.release()

        if ret:
            h, w = preview_frame.shape[:2]

            # Auto-align ROIs when video changes if preset is auto
            if cam_preset.startswith("🎯") and st.session_state.get("auto_aligned_video") != str(video_source_path):
                detected_mode, auto_str, auto_rt = auto_detect_signals(preview_frame)
                st.session_state["roi1_x1"], st.session_state["roi1_y1"] = auto_str[0], auto_str[1]
                st.session_state["roi1_x2"], st.session_state["roi1_y2"] = auto_str[2], auto_str[3]
                st.session_state["roi2_x1"], st.session_state["roi2_y1"] = auto_rt[0], auto_rt[1]
                st.session_state["roi2_x2"], st.session_state["roi2_y2"] = auto_rt[2], auto_rt[3]
                if detected_mode in ["12h_wide", "18h_wide"]:
                    st.session_state["line_x1"], st.session_state["line_y1"] = 0, 590
                    st.session_state["line_x2"], st.session_state["line_y2"] = 1550, 650
                else:
                    st.session_state["line_x1"], st.session_state["line_y1"] = 0, 520
                    st.session_state["line_x2"], st.session_state["line_y2"] = 1500, 750
                st.session_state["auto_aligned_video"] = str(video_source_path)
                st.rerun()

            col_btn1, col_btn2 = st.columns([1, 2])
            with col_btn1:
                if st.button("🎯 Tự Động Căn Đèn Theo Video (Auto-Snap)", help="Tự động dò tìm góc quay camera và căn đúng vị trí 2 cụm đèn"):
                    detected_mode, auto_str, auto_rt = auto_detect_signals(preview_frame)
                    st.session_state["roi1_x1"], st.session_state["roi1_y1"] = auto_str[0], auto_str[1]
                    st.session_state["roi1_x2"], st.session_state["roi1_y2"] = auto_str[2], auto_str[3]
                    st.session_state["roi2_x1"], st.session_state["roi2_y1"] = auto_rt[0], auto_rt[1]
                    st.session_state["roi2_x2"], st.session_state["roi2_y2"] = auto_rt[2], auto_rt[3]
                    if detected_mode in ["12h_wide", "18h_wide"]:
                        st.session_state["line_x1"], st.session_state["line_y1"] = 0, 590
                        st.session_state["line_x2"], st.session_state["line_y2"] = 1550, 650
                    else:
                        st.session_state["line_x1"], st.session_state["line_y1"] = 0, 520
                        st.session_state["line_x2"], st.session_state["line_y2"] = 1500, 750
                    st.session_state["auto_aligned_video"] = str(video_source_path)
                    st.success(f"Đã tự động căn đèn theo chế độ: `{detected_mode}`")
                    st.rerun()

            with col_btn2:
                show_grid = st.checkbox("📐 Bật lưới đo tọa độ (Coordinate Grid)", value=False, help="Vẽ lưới tọa độ mỗi 100px để bạn dễ dàng nhìn thấy tọa độ chính xác của mặt đường")

            # Detect current signal state on preview frame
            curr_str_roi = [int(roi1_x1), int(roi1_y1), int(roi1_x2), int(roi1_y2)]
            preview_straight_state = classify_approach_signal(preview_frame, curr_str_roi)

            def get_color_emoji(state):
                return "🟢" if state == "GREEN" else ("🔴" if state == "RED" else ("🟡" if state == "YELLOW" else "⚪"))

            if is_single_light_ui:
                st.info(
                    f"**Nhận diện tín hiệu đèn tại frame đầu tiên:** "
                    f"Đèn Tín Hiệu: {get_color_emoji(preview_straight_state)} `{preview_straight_state}` "
                    f"{'(Cho phép rẽ phải khi đèn đỏ)' if allow_right_on_red_ui else ''}"
                )
            else:
                curr_rt_roi = [int(roi2_x1), int(roi2_y1), int(roi2_x2), int(roi2_y2)]
                preview_right_state = classify_right_pole_assembly(preview_frame, curr_rt_roi)
                st.info(
                    f"**Nhận diện tín hiệu đèn tại frame đầu tiên:** "
                    f"Đèn Đi Thẳng: {get_color_emoji(preview_straight_state)} `{preview_straight_state}` | "
                    f"Đèn Rẽ Phải: {get_color_emoji(preview_right_state)} `{preview_right_state}`"
                )

            # Optional Grid Overlay
            if show_grid:
                grid_frame = preview_frame.copy()
                # Vertical grid lines
                for gx in range(0, w, 100):
                    color = (80, 80, 80) if gx % 200 != 0 else (120, 120, 120)
                    cv2.line(grid_frame, (gx, 0), (gx, h), color, 1)
                    if gx % 200 == 0:
                        cv2.putText(grid_frame, str(gx), (gx + 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Horizontal grid lines
                for gy in range(0, h, 100):
                    color = (80, 80, 80) if gy % 200 != 0 else (120, 120, 120)
                    cv2.line(grid_frame, (0, gy), (w, gy), color, 1)
                    if gy % 200 == 0:
                        cv2.putText(grid_frame, str(gy), (10, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                preview_frame = cv2.addWeighted(grid_frame, 0.75, preview_frame, 0.25, 0)

            # Draw Virtual Counting Line
            cv2.line(preview_frame, (int(line_x1), int(line_y1)), (int(line_x2), int(line_y2)), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(preview_frame, f"STOP LINE ({line_x1},{line_y1}) -> ({line_x2},{line_y2})", 
                        (max(10, int(line_x1) + 10), max(30, int(line_y1) - 12)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw Signal 1: Straight / Main Signal
            lbl = "TRAFFIC LIGHT" if is_single_light_ui else "STRAIGHT"
            cv2.rectangle(preview_frame, (int(roi1_x1), int(roi1_y1)), (int(roi1_x2), int(roi1_y2)), (0, 255, 255), 2)
            cv2.putText(preview_frame, f"{lbl}: {preview_straight_state}", (int(roi1_x1), max(20, int(roi1_y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if not is_single_light_ui:
                # Draw Signal 2: Right Turn Light ROI (Cyan box)
                cv2.rectangle(preview_frame, (int(roi2_x1), int(roi2_y1)), (int(roi2_x2), int(roi2_y2)), (255, 255, 0), 2)
                cv2.putText(preview_frame, f"RIGHT-TURN: {preview_right_state}", (max(10, int(roi2_x1) - 80), max(20, int(roi2_y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Display Preview
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption=f"Khung hình đầu tiên ({w}x{h}) - Xem trước Vạch dừng & Đèn Tín Hiệu", use_container_width=True)
        else:
            st.warning("Không thể đọc frame đầu tiên của video để xem trước.")


# =============================
# XỬ LÝ PHÂN TÍCH
# =============================
st.divider()
start_btn_bottom = st.button(
    "🚀 Bắt đầu Phân Tích Giao Thông (Phía Dưới)",
    type="primary",
    disabled=video_source_path is None,
    key="btn_start_bottom",
    use_container_width=True,
)
start_btn = start_btn_top or start_btn_bottom

if start_btn and video_source_path:
    # Prepare runtime config with Multi-Signal & Directional tracking
    run_config = base_config.copy()
    run_config["models"]["device"] = get_default_device(device_option)
    run_config["inference"]["confidence"] = conf_threshold
    run_config["counting_line"]["start"] = [int(line_x1), int(line_y1)]
    run_config["counting_line"]["end"] = [int(line_x2), int(line_y2)]
    if "tracking" not in run_config:
        run_config["tracking"] = {}
    run_config["tracking"]["map_road_riders_to_motorcycle"] = night_rider_boost
    if is_single_light_ui:
        run_config["traffic_signals"] = {
            "mode": "single",
            "has_right_turn_light": False,
            "allow_right_turn_on_red": allow_right_on_red_ui,
            "straight_light": {
                "name": "Đèn Tín Hiệu Chung",
                "roi": [int(roi1_x1), int(roi1_y1), int(roi1_x2), int(roi1_y2)],
                "min_pixels": 8,
            },
        }
        run_config["traffic_light"]["roi"] = [int(roi1_x1), int(roi1_y1), int(roi1_x2), int(roi1_y2)]
    else:
        run_config["traffic_signals"] = {
            "mode": "directional",
            "has_right_turn_light": True,
            "allow_right_turn_on_red": False,
            "straight_light": {
                "name": "Đèn Đi Thẳng",
                "roi": [int(roi1_x1), int(roi1_y1), int(roi1_x2), int(roi1_y2)],
                "min_pixels": 8,
            },
            "right_turn_light": {
                "name": "Đèn Rẽ Phải",
                "roi": [int(roi2_x1), int(roi2_y1), int(roi2_x2), int(roi2_y2)],
                "min_pixels": 6,
            },
        }
        run_config["traffic_light"]["roi"] = [int(roi1_x1), int(roi1_y1), int(roi1_x2), int(roi1_y2)]
    run_config["traffic_light"]["roi"] = [int(roi1_x1), int(roi1_y1), int(roi1_x2), int(roi1_y2)]

    output_dir = Path("outputs/ui_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # UI Progress containers
    progress_bar = st.progress(0, text="Khởi tạo mô hình và bộ theo dõi...")
    status_text = st.empty()
    metrics_placeholder = st.empty()

    def update_progress(current_frame, total_frames, current_signal, counts, violations):
        fraction = min(1.0, current_frame / max(1, total_frames))
        progress_bar.progress(
            fraction,
            text=f"Đang phân tích: {current_frame}/{total_frames} frames ({int(fraction * 100)}%) - Tín hiệu: {current_signal}"
        )
        status_text.caption(
            f"🚗 Car: {counts.get('car', 0)} | 🏍️ Moto: {counts.get('motorcycle', 0)} | "
            f"🚌 Bus: {counts.get('bus', 0)} | 🚚 Truck: {counts.get('truck', 0)} | 🚨 Vi phạm: {violations}"
        )

    try:
        with st.spinner("Đang tải mô hình YOLO và khởi chạy bộ theo dõi ByteTrack..."):
            pipeline = TrafficMonitorPipeline(run_config)

        summary = pipeline.run(
            video_source=str(video_source_path),
            output_dir=str(output_dir),
            save_video=save_video_out,
            save_csv=True,
            progress_callback=update_progress,
        )

        progress_bar.progress(1.0, text="Hoàn tất phân tích giao thông!")
        st.success("✅ Quá trình phân tích hoàn tất thành công!")

        # Display Metrics
        st.subheader("📊 Thống Kê Tổng Hợp")
        m_cols = st.columns(5)
        m_cols[0].metric("🚗 Ô tô (Car)", summary["counts"].get("car", 0))
        m_cols[1].metric("🏍️ Xe máy (Moto)", summary["counts"].get("motorcycle", 0))
        m_cols[2].metric("🚌 Xe buýt (Bus)", summary["counts"].get("bus", 0))
        m_cols[3].metric("🚚 Xe tải (Truck)", summary["counts"].get("truck", 0))
        m_cols[4].metric("🚨 Vượt đèn đỏ", summary["violations_count"])

        # Display Results in Tabs
        res_tab1, res_tab2, res_tab3 = st.tabs(["🎥 Video Kết Quả", "📋 Danh Sách Vi Phạm (CSV)", "📸 Bộ Sưu Tập Bằng Chứng"])

        with res_tab1:
            if summary["output_video"] and Path(summary["output_video"]).exists():
                st.video(summary["output_video"])
                with open(summary["output_video"], "rb") as vf:
                    st.download_button(
                        label="⬇️ Tải Video Kết Quả",
                        data=vf,
                        file_name="traffic_counted.mp4",
                        mime="video/mp4",
                    )
            else:
                st.info("Chế độ lưu video đã tắt hoặc không có video được ghi.")

        with res_tab2:
            if summary["output_csv"] and Path(summary["output_csv"]).exists():
                df = pd.read_csv(summary["output_csv"])
                st.dataframe(df, use_container_width=True)
                with open(summary["output_csv"], "rb") as cf:
                    st.download_button(
                        label="⬇️ Tải Báo Cáo CSV",
                        data=cf,
                        file_name="counted_vehicles.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Không có dữ liệu vi phạm nào.")

        with res_tab3:
            events = summary.get("events", [])
            if events:
                st.write(f"Tìm thấy **{len(events)}** sự kiện vi phạm:")
                for idx, event in enumerate(events):
                    ev_img_path = Path(event.get("evidence_image", ""))
                    if not ev_img_path.exists():
                        continue
                    
                    st.divider()
                    col_img, col_info = st.columns([3, 2])
                    with col_img:
                        st.image(
                            str(ev_img_path),
                            caption=f"Hình ảnh bằng chứng vi phạm - Sự kiện #{event['event_id']} (Frame {event['frame']})",
                            use_container_width=True,
                        )
                    with col_info:
                        st.markdown(f"#### 📋 Biên Bản Vi Phạm #{event['event_id']}")
                        st.write(f"• **Mã đối tượng (Track ID):** `{event['track_id']}`")
                        st.write(f"• **Loại phương tiện:** `{event['vehicle_type'].upper()}`")
                        st.write(f"• **Lỗi ghi nhận:** `{event.get('violation_type', 'RED_LIGHT').upper()}`")
                        st.write(f"• **Hướng di chuyển:** `{event.get('movement_direction', 'N/A').upper()}`")
                        st.write(f"• **Tín hiệu đèn:** `{event.get('signal_responsible', 'signal')} = {event.get('signal_state', 'RED')}`")
                        st.write(f"• **Thời điểm vi phạm:** Frame `{event['frame']}`")

                        curr_bks = str(event.get("license_plate", "")).strip()
                        is_unclear = not curr_bks or "chưa rõ" in curr_bks.lower() or "id_" in curr_bks.lower()
                        init_val = "" if is_unclear else curr_bks

                        st.markdown("**🔍 Xác nhận / Chỉnh sửa Biển Kiểm Soát (BKS):**")
                        user_bks = st.text_input(
                            f"Nhập BKS cho sự kiện #{event['event_id']}:",
                            value=init_val,
                            placeholder="Ví dụ: 27-B2 233.80",
                            key=f"bks_input_{event['event_id']}_{idx}",
                        )
                        if st.button(f"💾 Lưu BKS vào biên bản & CSV", key=f"btn_save_bks_{event['event_id']}_{idx}"):
                            confirmed_bks = user_bks.strip().upper()
                            if confirmed_bks:
                                event["license_plate"] = confirmed_bks
                                # Update CSV file
                                csv_path = Path(summary["output_csv"])
                                if csv_path.exists():
                                    try:
                                        df_csv = pd.read_csv(csv_path)
                                        df_csv.loc[df_csv["event_id"] == event["event_id"], "license_plate"] = confirmed_bks
                                        df_csv.to_csv(csv_path, index=False)
                                    except Exception as e_csv:
                                        st.warning(f"Lỗi khi cập nhật file CSV: {e_csv}")

                                # Update evidence image overlay if possible
                                try:
                                    v_crop_path = Path(event.get("vehicle_crop", ""))
                                    if v_crop_path.exists():
                                        v_crop = cv2.imread(str(v_crop_path))
                                        full_ev = cv2.imread(str(ev_img_path))
                                        # Left side of full_ev is the violation scene
                                        if full_ev is not None:
                                            # Crop left scene portion
                                            scene_w = max(1, int(full_ev.shape[1] * 0.58))
                                            scene_frame = full_ev[:, :scene_w]
                                            p_zoom, _ = enhance_license_plate(v_crop, zoom_factor=4)
                                            updated_ev = create_combined_evidence(
                                                scene_frame,
                                                plate_zoom=p_zoom,
                                                plate_text=confirmed_bks,
                                                vehicle_crop=v_crop,
                                            )
                                            cv2.imwrite(str(ev_img_path), updated_ev)
                                except Exception:
                                    pass

                                st.success(f"✅ Đã xác nhận BKS: **{confirmed_bks}** cho sự kiện #{event['event_id']}!")
                                st.rerun()
            else:
                st.info("Không ghi nhận vi phạm vượt đèn đỏ nào trong video.")

    except Exception as err:
        st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {err}")
