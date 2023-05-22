docker build -t bmi_predictor:v1 .

docker run -p 8501:8501 bmi_predictor:v1
