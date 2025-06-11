import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd

try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.info("Please install opencv-python-headless for deployment environments")
    st.stop()
import numpy as np
from datetime import datetime


# 🔐 OpenAI API 키

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def get_features(image: Image.Image):
    img = np.array(image)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Sobel 필터로 엣지(구조) 추출
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobelx**2 + sobely**2)

    # 2. 색상 특징 추출 (RGB 평균)
    mean_color = img.mean(axis=(0, 1))  # shape (3,)

    # 3. 벡터로 결합 (flatten된 edge + 색상 평균)
    edge_vec = cv2.resize(edge, (32, 32)).flatten()  # 줄여서 벡터화
    edge_vec = edge_vec / np.linalg.norm(edge_vec)  # 정규화
    color_vec = mean_color / 255
    feature_vec = np.concatenate([edge_vec, color_vec])

    return feature_vec


def compute_similarity(
    created_image_tensor: torch.Tensor,
    target_image_tensor: torch.Tensor,
):
    model = models.vgg16(pretrained=True)
    model.eval()
    with torch.no_grad():
        created_image_feature = model(created_image_tensor)
        target_image_feature = model(target_image_tensor)
    return float(
        (created_image_feature @ target_image_feature.T).item()
        / (np.linalg.norm(created_image_feature) * np.linalg.norm(target_image_feature))
    )


def update_leaderboard(name, score, file_name):
    file = "leaderboard.csv"
    df = (
        pd.read_csv(file)
        if os.path.exists(file)
        else pd.DataFrame(columns=["이름", "점수", "사진"])
    )
    df.loc[len(df)] = [name, round(score, 4), file_name]
    df.to_csv(file, index=False)
    return df


def generate_image_from_openai(username, prompt):
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=1,
        size="512x512",
    )
    image_url = response.data[0].url
    created = response.created
    image = Image.open(BytesIO(requests.get(image_url).content))
    # 현재 시간과 날짜 가져오기
    current_time = datetime.now()
    # 월-일-시간-분 형태로 포맷팅
    formatted_time = current_time.strftime("%m-%d-%H-%M")

    # generated_images 폴더가 없으면 생성
    os.makedirs("generated_images", exist_ok=True)
    file_name = f"{username}_{formatted_time}_{created}"
    image.save(f"generated_images/{file_name}")
    return image, file_name


# 1. Define the model
class PretrainedResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        base_model = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # remove avgpool + fc
        self.projector = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        x = self.projector(x)  # [B, output_dim]
        return x


# 2. Define the image transform (resize + ToTensor only, no normalization)
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # scales to [0, 1]
    ]
)


# 3. Load image (PIL.Image)
def load_image(image: Image.Image):
    img = image.convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dimension → [1, 3, 512, 512]


# 세션 스테이트 초기화
if "selected_leaderboard_image" not in st.session_state:
    st.session_state.selected_leaderboard_image = None

# 🌐 Streamlit UI
st.title("이미지 프롬프팅 챌린지")
st.subheader("주어진 사진과 가장 유사한 사진을 만드신분에게 소정의 상품을 드립니다.")
st.info("기한 : 2025년 6월 13일 18시 00분")
st.image("images/target_image.png", caption="목표 사진", use_container_width=True)

name = st.text_input("👤 이름을 입력하세요(학번과 이름)")
prompt = st.text_area("💬 이미지를 생성할 프롬프트를 입력하세요")


@st.dialog("이미지 보기")
def show_image(image, score):
    try:
        st.image(f"generated_images/{image}.png", caption=f"유사도 : {score}")
    except Exception as e:
        st.error(f"이미지를 불러올 수 없습니다. {e}")
    if st.button("확인", type="primary"):
        st.session_state.selected_leaderboard_image = None
        st.rerun()


if st.session_state.selected_leaderboard_image:
    show_image(
        st.session_state.selected_leaderboard_image["url"],
        st.session_state.selected_leaderboard_image["score"],
    )

with st.sidebar:
    st.subheader("🏆 리더보드 (Top 10)")

    # 리더보드 파일이 있는지 확인
    if os.path.exists("leaderboard.csv"):
        df = pd.read_csv("leaderboard.csv")
        df = df.sort_values("점수", ascending=False).reset_index(drop=True)[:10]

        if not df.empty:
            for idx, row in df.iterrows():
                rank = idx + 1
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**{rank}위** {row['이름']}")
                    st.write(f"점수: {row['점수']}")

                with col2:
                    if st.button("사진보기", key=f"view_{row['사진']}"):
                        st.session_state.selected_leaderboard_image = {
                            "name": row["이름"],
                            "score": row["점수"],
                            "url": row["사진"],
                        }
                        st.rerun()

                st.markdown("---")
        else:
            st.info("아직 참여자가 없습니다.")
    else:
        st.info("아직 참여자가 없습니다.")

    st.info("페이지를 새로고침하면 리더보드가 업데이트됩니다.")


if st.button("이미지 생성 및 유사도 계산"):
    if not name or not prompt:
        st.warning("이름과 프롬프트를 모두 입력해주세요.")
    else:
        with st.spinner("이미지 생성 중..."):
            try:
                generated_img, file_name = generate_image_from_openai(name, prompt)
            except Exception as e:
                st.error(f"이미지 생성 실패: {e}")
                st.stop()

        st.image(generated_img, caption="🖼️ 생성된 이미지", use_container_width=True)

        with st.spinner("유사도 계산 중..."):
            scores = 0.0
            generated_img_tensor = load_image(generated_img)
            target_img_tensor = load_image(Image.open("images/target_image.png"))
            scores = compute_similarity(generated_img_tensor, target_img_tensor)
            st.success(f"유사도 점수: {scores:.4f}")

        df = update_leaderboard(name, scores, file_name)
