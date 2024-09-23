FROM python:3.11-slim
WORKDIR /code
COPY ./ /code
RUN pip install .
RUN pip install anthropic

EXPOSE 7860

CMD ["python", "-m", "rag_chatbot", "--host", "host.docker.internal"]