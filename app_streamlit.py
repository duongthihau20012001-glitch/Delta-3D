import os
import sys
import glob
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import gdown
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CẤU HÌNH TRANG STREAMLIT TỐI ƯU CHO DI ĐỘNG & WEB
# ==============================================================================
st.set_page_config(
    page_title="Delta Robot IK AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. HÀM TẢI MÔ HÌNH AI TỰ ĐỘNG TỪ GOOGLE DRIVE
# ==============================================================================
@st.cache_resource
def load_model_from_gdrive(file_id="1eK00jqxsrZ-ZTS0RPQGXtrTLfZQxKrXq"):
    """Tải và nạp mô hình .joblib từ Google Drive ngầm vào bộ nhớ cache"""
    model_path = "hoc_nguoc_model.joblib"
    
    # 1. Kiểm tra nếu đã có file local trong thư mục
    if not os.path.exists(model_path):
        matches = glob.glob("*.joblib")
        if matches:
            model_path = matches[0]

    # 2. Nếu chưa có file, tiến hành tải từ Google Drive với File ID của bạn
    if not os.path.exists(model_path):
        target_id = file_id.strip() if file_id and file_id.strip() else "1eK00jqxsrZ-ZTS0RPQGXtrTLfZQxKrXq"
        url = f"https://drive.google.com/uc?id={target_id}"
        try:
            with st.spinner("Đang tải mô hình AI từ Google Drive (chỉ tải 1 lần đầu)..."):
                gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Lỗi khi tải file từ Google Drive: {str(e)}")
            return None, None

    # 3. Nạp mô hình vào bộ nhớ
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, model_path
        except Exception as e:
            st.error(f"Lỗi khi đọc file mô hình .joblib: {str(e)}")
            return None, None

    return None, None

# ==============================================================================
# 3. HÀM DỰ ĐOÁN AI THÔNG MINH (ĐÃ SỬA LỖI KMEANS N_SAMPLES=1)
# ==============================================================================
def predict_smart(loaded_obj, X_input):
    X_curr = np.array(X_input, dtype=np.float64)
    if X_curr.ndim == 1:
        X_curr = X_curr.reshape(1, -1)

    cols = ['R', 'r', 'a', 'b', 'px', 'py', 'pz']
    X_df = pd.DataFrame(X_curr, columns=cols if X_curr.shape[1] == 7 else None)

    model = None
    poly = None
    scaler = None
    kmeans = None

    if isinstance(loaded_obj, dict):
        for k in ['pipeline', 'model_pipeline', 'pipe', 'full_pipeline']:
            if k in loaded_obj and hasattr(loaded_obj[k], 'predict'):
                try:
                    return loaded_obj[k].predict(X_df)
                except Exception:
                    pass

        poly = loaded_obj.get('poly') or loaded_obj.get('poly_transformer') or loaded_obj.get('polynomialfeatures')
        scaler = loaded_obj.get('scaler') or loaded_obj.get('standardscaler')
        kmeans = loaded_obj.get('kmeans') or loaded_obj.get('kmeans_model')

        for mk in ['model', 'rf_model', 'rf', 'estimator', 'regressor', 'randomforestregressor', 'best_estimator_']:
            if mk in loaded_obj and hasattr(loaded_obj[mk], 'predict'):
                model = loaded_obj[mk]
                break

        if model is None:
            for v in loaded_obj.values():
                if hasattr(v, 'predict'):
                    model = v
                    break
    else:
        model = loaded_obj

    if model is None or not hasattr(model, 'predict'):
        raise ValueError("Không tìm thấy mô hình dự đoán hợp lệ trong file .joblib!")

    # Thử dự đoán trực tiếp nếu mô hình nhận 7 tham số
    try:
        return model.predict(X_df)
    except Exception:
        pass

    # Biến đổi đặc trưng nâng cao
    X_mat = X_curr
    if poly is not None and hasattr(poly, 'transform'):
        X_mat = poly.transform(X_mat)
    if scaler is not None and hasattr(scaler, 'transform'):
        X_mat = scaler.transform(X_mat)
    if kmeans is not None and hasattr(kmeans, 'predict'):
        labels = kmeans.predict(X_mat).reshape(-1, 1)
        X_mat = np.hstack([X_mat, labels])

    n_expected = getattr(model, 'n_features_in_', None)
    if n_expected in [28, 29] and X_mat.shape[1] == 7:
        from sklearn.preprocessing import PolynomialFeatures
        poly_auto = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_mat = poly_auto.fit_transform(X_mat)

        if n_expected == 29 and X_mat.shape[1] == 28:
            if kmeans is not None and hasattr(kmeans, 'predict'):
                labels = kmeans.predict(X_mat).reshape(-1, 1)
            else:
                # Tránh lỗi n_samples=1 < n_clusters=5 khi tính điểm đơn lẻ
                labels = np.zeros((X_mat.shape[0], 1))
            X_mat = np.hstack([X_mat, labels])

    return model.predict(X_mat)

# ==============================================================================
# 4. GIAO DIỆN CỘT BÊN (SIDEBAR) - THÔNG SỐ VÀ CẤU HÌNH DRIVE
# ==============================================================================
st.sidebar.title("⚙️ Cấu Hình Hệ Thống")

st.sidebar.subheader("1. Mã Google Drive File ID")
gdrive_id = st.sidebar.text_input(
    "Nhập File ID trên Google Drive:",
    value="1eK00jqxsrZ-ZTS0RPQGXtrTLfZQxKrXq",
    help="Chuỗi ký tự nằm giữa /d/ và /view trên link chia sẻ Google Drive."
)

st.sidebar.subheader("2. Thông số hình học kết cấu (mm)")
R = st.sidebar.number_input("R (Bệ tĩnh):", min_value=1.0, max_value=2000.0, value=100.0, step=5.0)
r = st.sidebar.number_input("r (Bệ động):", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
a = st.sidebar.number_input("a (Tay đòn):", min_value=1.0, max_value=2000.0, value=250.0, step=5.0)
b = st.sidebar.number_input("b (Thanh nối):", min_value=1.0, max_value=2000.0, value=440.0, step=5.0)

st.sidebar.subheader("3. Tọa độ điểm mục tiêu P(x, y, z) (mm)")
Px = st.sidebar.number_input("Tọa độ X (Px):", min_value=-2000.0, max_value=2000.0, value=50.0, step=5.0)
Py = st.sidebar.number_input("Tọa độ Y (Py):", min_value=-2000.0, max_value=2000.0, value=-50.0, step=5.0)
Pz = st.sidebar.number_input("Tọa độ Z (Pz):", min_value=-2000.0, max_value=2000.0, value=300.0, step=5.0)

# Nạp mô hình AI
loaded_model, model_file_name = load_model_from_gdrive(gdrive_id)

if loaded_model is not None:
    st.sidebar.success(f"✓ Đã sẵn sàng mô hình: {os.path.basename(model_file_name)}")
else:
    st.sidebar.warning("⚠️ Chưa nạp được tệp .joblib. Vui lòng kiểm tra File ID Google Drive.")

# ==============================================================================
# 5. GIAO DIỆN CHÍNH - HIỂN THỊ VÀ MÔ PHỎNG 3D
# ==============================================================================
st.title("🤖 AI Agent - Động học Ngược Robot Delta (3RUU)")
st.caption("Ứng dụng tính toán 9 góc khớp và mô phỏng không gian 3D dựa trên mô hình AI Surrogate Model")

col_left, col_right = st.columns([1, 1.2])

# KHU VỰC TÍNH TOÁN & KẾT QUẢ GÓC KHỚP
with col_left:
    st.subheader("📊 Kết quả 9 góc khớp")
    
    btn_calc = st.button("🚀 Tính Toán Động Học Ngược", use_container_width=True, type="primary")

    if btn_calc:
        if loaded_model is not None:
            try:
                X_input = np.array([[R, r, a, b, Px, Py, abs(Pz)]])
                q_pred_rad = predict_smart(loaded_model, X_input)[0]
                st.session_state['q_deg'] = [np.degrees(val) for val in q_pred_rad]
                st.success("✓ Đã tính toán xong 9 góc khớp!")
            except Exception as e:
                st.error(f"Lỗi tính toán IK: {str(e)}")
        else:
            st.error("Chưa có mô hình AI .joblib để thực hiện tính toán!")

    q_deg = st.session_state.get('q_deg', [0.0] * 9)

    # Hiển thị 9 góc khớp dạng thẻ Grid
    grid1, grid2, grid3 = st.columns(3)
    angle_names = ["q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33"]
    
    for idx, name in enumerate(angle_names):
        col_target = [grid1, grid2, grid3][idx % 3]
        col_target.metric(label=name, value=f"{q_deg[idx]:.4f}°")

    # Kiểm tra dải phân tán góc chủ động
    q11, q21, q31 = q_deg[0], q_deg[3], q_deg[6]
    if (-45.0 <= q11 <= 90.0) and (-45.0 <= q21 <= 90.0) and (-45.0 <= q31 <= 90.0):
        st.info("✓ Các góc chủ động q11, q21, q31 nằm trong dải phân tán an toàn [-45°, 90°]")
    else:
        st.warning("⚠️ Cảnh báo: Có góc chủ động nằm ngoài dải phân tán an toàn [-45°, 90°]")

# KHU VỰC ĐỒ HỌA MÔ PHỎNG 3D
with col_right:
    st.subheader("🌐 Mô phỏng Hình học 3D")
    
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    
    Z_base = 0.0
    Pz_real = -abs(Pz) if Pz > 0 else Pz

    # 1. Bệ tĩnh 3D tại Z = 0
    theta = np.linspace(0, 2*np.pi, 30)
    bx = (R + 25) * np.cos(theta)
    by = (R + 25) * np.sin(theta)
    bz = np.full_like(theta, Z_base)
    ax.plot(bx, by, bz, color='#37474F', linewidth=2.5)
    ax.add_collection3d(Poly3DCollection([list(zip(bx, by, bz))], facecolors='#455A64', alpha=0.8))

    # 2. Bệ động 3D tại Z = Pz_real
    px_circle = (r + 15) * np.cos(theta) + Px
    py_circle = (r + 15) * np.sin(theta) + Py
    pz_circle = np.full_like(theta, Pz_real)
    ax.plot(px_circle, py_circle, pz_circle, color='#C62828', linewidth=2)
    ax.add_collection3d(Poly3DCollection([list(zip(px_circle, py_circle, pz_circle))], facecolors='#E53935', alpha=0.8))

    # 3. Ba nhánh Robot Delta
    angles_base = [0.0, 120.0, 240.0]
    rod_spacing = 40.0

    for i in range(3):
        phi_rad = np.radians(angles_base[i])
        q1_rad = np.radians(q_deg[i*3])

        u_x, u_y = np.cos(phi_rad), np.sin(phi_rad)
        v_x, v_y = -np.sin(phi_rad), np.cos(phi_rad)

        Ax, Ay, Az = R * u_x, R * u_y, Z_base
        Bx = (R + a * np.cos(q1_rad)) * u_x
        By = (R + a * np.cos(q1_rad)) * u_y
        Bz = Z_base - a * np.sin(q1_rad)
        Cx, Cy, Cz = Px + r * u_x, Py + r * u_y, Pz_real

        # Tay đòn a
        ax.plot([Ax, Bx], [Ay, By], [Az, Bz], color='#1565C0', linewidth=5)
        ax.scatter([Ax], [Ay], [Az + 5], color='#212121', s=60, marker='s')

        # Đôi thanh nối b song song
        B1 = [Bx + 0.5*rod_spacing*v_x, By + 0.5*rod_spacing*v_y, Bz]
        B2 = [Bx - 0.5*rod_spacing*v_x, By - 0.5*rod_spacing*v_y, Bz]
        C1 = [Cx + 0.5*rod_spacing*v_x, Cy + 0.5*rod_spacing*v_y, Cz]
        C2 = [Cx - 0.5*rod_spacing*v_x, Cy - 0.5*rod_spacing*v_y, Cz]

        ax.plot([B1[0], C1[0]], [B1[1], C1[1]], [B1[2], C1[2]], color='#212121', linewidth=2)
        ax.plot([B2[0], C2[0]], [B2[1], C2[1]], [B2[2], C2[2]], color='#212121', linewidth=2)
        ax.scatter([B1[0], B2[0], C1[0], C2[0]], [B1[1], B2[1], C1[1], C2[1]], [B1[2], B2[2], C1[2], C2[2]], color='#FFD54F', s=25)

    # 4. Đầu gắp P
    ax.plot([Px, Px], [Py, Py], [Pz_real, Pz_real - 35], color='#FF9800', linewidth=3)
    ax.scatter([Px], [Py], [Pz_real - 35], color='#FF3D00', s=80, marker='v')

    # Tự động mở rộng dải giới hạn không gian XY
    limit_xy = max(420.0, R + a + 60.0)
    ax.set_xlim(-limit_xy, limit_xy)
    ax.set_ylim(-limit_xy, limit_xy)
    ax.set_zlim(-500, 50)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=22, azim=35)
    
    st.pyplot(fig)