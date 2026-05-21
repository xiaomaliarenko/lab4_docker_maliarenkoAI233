# Модель: Оптимальне керування процесом очищення водойми (5 семестр)
# Автор: Маляренко Анастасія (у групі з Брагар Софією), група АІ-233

FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy
COPY main.py .
CMD ["python", "main.py"]
