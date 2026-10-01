
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 자동차 연비 데이터 불러오기
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

df = pd.read_csv(url)
data = df[["weight", "mpg"]].dropna()

X = data[["weight"]]
y = data["mpg"]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 선형회귀 모델
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 비선형회귀 모델
poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model.fit(X_train, y_train)

# 모델 성능 평가
linear_pred = linear_model.predict(X_test)
poly_pred = poly_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
poly_r2 = r2_score(y_test, poly_pred)

# 자동차 연비 예측 함수
def predict_mpg(weight, model_type):

    if weight <= 0:
        return "자동차 무게는 0보다 큰 숫자를 입력해 주세요."

    if model_type == "선형회귀":
        prediction = linear_model.predict([[weight]])[0]
    else:
        prediction = poly_model.predict([[weight]])[0]

    return f"예측 자동차 연비: {prediction:.2f} MPG"

# 웹 프로그램 화면
demo = gr.Interface(
    fn=predict_mpg,
    inputs=[
        gr.Number(
            label="자동차 무게 (파운드)",
            value=3000
        ),
        gr.Radio(
            choices=["선형회귀", "비선형회귀"],
            label="회귀 모델 선택",
            value="선형회귀"
        )
    ],
    outputs=gr.Textbox(label="예측 결과"),
    title="자동차 연비 예측 프로그램",
    description=(
        "선형회귀와 비선형회귀를 이용하여 "
        "자동차 무게에 따른 연비를 예측합니다."
    )
)

if __name__ == "__main__":
    demo.launch()
