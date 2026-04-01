import base64
import io
import json
import re
import uuid

import boto3
import streamlit as st
from PIL import Image

# ===== CONFIG FROM STREAMLIT SECRETS =====
AWS_REGION = st.secrets["AWS_REGION"]
RUNTIME_ARN = st.secrets["RUNTIME_ARN"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

DEFAULT_CONF = 0.25


@st.cache_resource
def get_agentcore_client():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    return session.client("bedrock-agentcore")


def image_to_base64(uploaded_file) -> str:
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


def make_runtime_session_id(filename: str) -> str:
    """
    AgentCore yêu cầu runtimeSessionId chỉ gồm chữ và số
    và đủ dài. Tạo ID dài 33-64 ký tự.
    """
    base = re.sub(r"[^a-zA-Z0-9]", "", filename)
    unique = uuid.uuid4().hex
    session_id = f"{base}{unique}"

    if len(session_id) < 33:
        session_id += "x" * (33 - len(session_id))

    return session_id[:64]


def invoke_agentcore(
    client,
    uploaded_file,
    conf: float,
    show_reference: bool,
    response_mode: str = "json",
):
    payload = {
        "filename": uploaded_file.name,
        "image_base64": image_to_base64(uploaded_file),
        "conf": conf,
        "show_reference": show_reference,
        "include_annotated_base64": True,   # luôn lấy ảnh annotated
        "response_mode": response_mode,
    }

    resp = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        runtimeSessionId=make_runtime_session_id(uploaded_file.name),
        payload=json.dumps(payload).encode("utf-8"),
        qualifier="DEFAULT",
    )

    raw = resp["response"].read().decode("utf-8")
    return json.loads(raw)


def decode_annotated_image(result: dict):
    annotated_b64 = result.get("annotated_image_base64")
    if not annotated_b64:
        return None

    image_bytes = base64.b64decode(annotated_b64)
    return Image.open(io.BytesIO(image_bytes))


def main():
    st.set_page_config(page_title="AgentCore Inspection", layout="wide")
    st.title("AgentCore kiểm tra vị trí sản phẩm")

    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )

    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF, 0.05)
    show_reference = st.checkbox("Hiện vùng chuẩn", value=True)

    if uploaded_files:
        client = get_agentcore_client()
        results_all = []

        with st.spinner("Đang gửi nhiều ảnh lên AgentCore..."):
            for uploaded_file in uploaded_files:
                try:
                    result = invoke_agentcore(
                        client=client,
                        uploaded_file=uploaded_file,
                        conf=conf,
                        show_reference=show_reference,
                        response_mode="json",
                    )
                    result["filename"] = uploaded_file.name
                    results_all.append(result)
                except Exception as e:
                    results_all.append(
                        {
                            "filename": uploaded_file.name,
                            "status": "ERROR",
                            "message": str(e),
                            "detections": {},
                            "missing": [],
                            "shifted": [],
                            "size_abnormal": [],
                        }
                    )

        st.subheader("Tổng hợp kết quả")
        ok_count = sum(1 for x in results_all if x.get("status") == "OK")
        ng_count = sum(1 for x in results_all if x.get("status") == "NG")
        err_count = sum(1 for x in results_all if x.get("status") == "ERROR")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.success(f"OK: {ok_count}")
        with col_b:
            st.error(f"NG: {ng_count}")
        with col_c:
            st.warning(f"ERROR: {err_count}")

        for item in results_all:
            st.markdown("---")
            st.subheader(item.get("filename", "unknown"))

            col1, col2 = st.columns([2, 1])

            with col1:
                annotated_img = decode_annotated_image(item)
                if annotated_img is not None:
                    st.image(
                        annotated_img,
                        caption=item.get("filename", ""),
                        use_container_width=True,
                    )
                else:
                    st.info("Không nhận được ảnh annotated từ AgentCore.")

            with col2:
                status = item.get("status", "UNKNOWN")

                if status == "OK":
                    st.success("OK")
                elif status == "NG":
                    st.error("NG")
                else:
                    st.warning("ERROR")

                st.write(f"**Message:** {item.get('message', '')}")

                st.write("**Part detect được:**")
                detections = item.get("detections", {})
                if detections:
                    for class_name, d in detections.items():
                        st.write(
                            f"- {class_name}: conf={d['conf']:.2f}, "
                            f"cx={d['cx']:.1f}, cy={d['cy']:.1f}"
                        )
                else:
                    st.write("- Không detect được part nào")

                if item.get("missing"):
                    st.write("**Thiếu part:**")
                    for x in item["missing"]:
                        st.write(f"- {x}")

                if item.get("shifted"):
                    st.write("**Lệch vị trí:**")
                    for x in item["shifted"]:
                        st.write(f"- {x}")

                if item.get("size_abnormal"):
                    st.write("**Kích thước box bất thường:**")
                    for x in item["size_abnormal"]:
                        st.write(f"- {x}")


if __name__ == "__main__":
    main()
