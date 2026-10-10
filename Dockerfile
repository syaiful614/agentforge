FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd -m agent && mkdir /workspace && chown agent /workspace
USER agent
ENV AGENTFORGE_WORKSPACE=/workspace
ENTRYPOINT ["agentforge"]
