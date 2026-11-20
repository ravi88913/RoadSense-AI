
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from model import load_model, predict_image
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


st.set_page_config(
    page_title="RoadSense AI",
    page_icon="🚗",
    layout="wide"
)

st.title("RoadSense AI")
st.subheader("Vision-Based Road Segmentation and Path Guidance")


@st.cache_resource
def get_model():
    return load_model()


def is_road_detected(prediction):
    h, w = prediction.shape

    road_region = prediction[
        int(h * 0.60):int(h * 0.95),
        int(w * 0.20):int(w * 0.80)
    ]

    if road_region.size == 0:
        return False

    road_ratio = np.mean(road_region == 0)

    return road_ratio >= 0.30


def get_obstacle_aware_mask(prediction):
    road_mask = (prediction == 0).astype(np.uint8)

    obstacle_mask = np.isin(
        prediction,
        [8, 9, 10]
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (25, 25)
    )

    obstacle_mask = cv2.dilate(
        obstacle_mask,
        kernel,
        iterations=1
    )

    safe_road_mask = road_mask & (obstacle_mask == 0)

    return safe_road_mask * 255


def draw_road_path(image, road_mask):
    h, w = road_mask.shape
    points = []

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    clean_mask = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    previous_center = None

    for y in np.linspace(
        int(h * 0.95),
        int(h * 0.45),
        40
    ).astype(int):

        row = clean_mask[y] > 0
        indices = np.where(row)[0]

        if len(indices) < w * 0.08:
            continue

        splits = np.where(np.diff(indices) > 1)[0] + 1
        groups = np.split(indices, splits)

        groups = [
            group for group in groups
            if len(group) >= w * 0.04
        ]

        if not groups:
            continue

        candidates = []

        for group in groups:
            center = int((group[0] + group[-1]) / 2)
            candidates.append((center, group))

        if previous_center is None:
            selected = min(
                candidates,
                key=lambda item: abs(item[0] - w * 0.5)
            )
        else:
            selected = min(
                candidates,
                key=lambda item: abs(item[0] - previous_center)
            )

        center = selected[0]
        previous_center = center

        points.append((center, y))

    result = image.copy()

    if len(points) < 8:
        return result, False

    points = np.array(points, dtype=np.float32)

    x_values = points[:, 0]
    y_values = points[:, 1]

    try:
        coefficients = np.polyfit(
            y_values,
            x_values,
            2
        )

        smooth_y = np.linspace(
            min(y_values),
            max(y_values),
            100
        )

        smooth_x = np.polyval(
            coefficients,
            smooth_y
        )

        if np.any(smooth_x < 0) or np.any(smooth_x >= w):
            return result, False

        smooth_points = np.column_stack(
            (smooth_x, smooth_y)
        ).astype(np.int32)

        cv2.polylines(
            result,
            [smooth_points],
            False,
            (255, 0, 255),
            5
        )

        return result, True

    except (ValueError, np.linalg.LinAlgError):
        return result, False


def create_road_overlay(image, road_mask):
    overlay = image.copy()

    overlay[road_mask > 0] = (
        0.55 * overlay[road_mask > 0]
        + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)

    return overlay


class RoadVideoProcessor(VideoProcessorBase):

    def __init__(self, network):
        self.network = network
        self.lock = threading.Lock()

    def recv(self, frame):

        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        output = frame_bgr.copy()
        inference_time = 0.0

        try:
            with self.lock:
                pil_image = Image.fromarray(frame_rgb)

                prediction, inference_time = predict_image(
                    self.network,
                    pil_image
                )

            prediction = np.asarray(prediction)

            if prediction.shape != frame_rgb.shape[:2]:
                prediction = cv2.resize(
                    prediction.astype(np.uint8),
                    (frame_rgb.shape[1], frame_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            if not is_road_detected(prediction):

                cv2.putText(
                    output,
                    "ROAD NOT FOUND",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    output,
                    "Please point camera towards the road",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            else:

                safe_mask = get_obstacle_aware_mask(
                    prediction
                )

                output, path_found = draw_road_path(
                    frame_bgr,
                    safe_mask
                )

                if path_found:
                    status = "PATH DETECTED"
                    color = (0, 255, 0)
                else:
                    status = "NO CLEAR PATH"
                    color = (0, 0, 255)

                cv2.putText(
                    output,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

            cv2.putText(
                output,
                f"Inference: {inference_time:.3f}s",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24"
            )

        except Exception:
            cv2.putText(
                output,
                "PROCESSING ERROR",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24"
            )


try:
    network = get_model()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Mode",
    ["Home", "Image Analysis", "Live Camera"]
)


if page == "Home":

    st.title("Welcome to RoadSense AI")

    st.write(
        "A deep learning based road segmentation and "
        "path guidance prototype using U-Net with ResNet34."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Architecture", "U-Net")
    col2.metric("Encoder", "ResNet34")
    col3.metric("Classes", "12")

    st.info(
        "Select Image Analysis or Live Camera from the sidebar."
    )


elif page == "Image Analysis":

    st.header("Road Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Road Image",
            use_container_width=True
        )

        if st.button("Analyze Image", type="primary"):

            with st.spinner("Analyzing road image..."):

                prediction, inference_time = predict_image(
                    network,
                    image
                )

                prediction = np.asarray(prediction)

                original = np.array(image)

                if prediction.shape != original.shape[:2]:
                    prediction = cv2.resize(
                        prediction.astype(np.uint8),
                        (original.shape[1], original.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                if not is_road_detected(prediction):

                    st.warning(
                        "Road not found. Please upload a clear road image."
                    )

                    st.stop()

                road_mask = get_obstacle_aware_mask(
                    prediction
                )

                overlay = create_road_overlay(
                    original,
                    road_mask
                )

                path_image, path_found = draw_road_path(
                    original,
                    road_mask
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        original,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        overlay,
                        caption="Road Segmentation",
                        use_container_width=True
                    )

                st.image(
                    road_mask,
                    caption="Obstacle-Aware Road Mask",
                    use_container_width=True
                )

                st.image(
                    path_image,
                    caption="Road Path Guidance",
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Inference Time",
                    f"{inference_time:.3f} seconds"
                )

                if path_found:
                    col2.success("Road Path Detected")
                else:
                    col2.warning("No Clear Path Detected")


elif page == "Live Camera":

    st.header("Live Camera")

    st.write(
        "Real-time road segmentation and path guidance using your webcam."
    )

    webrtc_streamer(
        key="roadsense-live-camera",
        video_processor_factory=lambda: RoadVideoProcessor(network),
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    st.caption(
        "Allow camera permission in your browser to start the live feed."
    )
