FROM 10.207.28.157:30080/infra/python-amd64:3.11 as builder
# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

COPY requirements.txt .
# 安装系统依赖（添加重试机制）
RUN for i in $(seq 1 3); do \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ build-essential python3-dev libc6-dev curl && \
    rm -rf /var/lib/apt/lists/* && \
    break || sleep 15; \
    done

# 复制项目文件
COPY inference.py .
COPY timefeatures.py .
#COPY README.md .
COPY TimesNet.py .
COPY layers/ ./layers/

# 升级pip并使用阿里云镜像源安装Python依赖
RUN pip install --upgrade pip && \
    pip install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r requirements.txt &&\
    pip install --no-cache-dir torch --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu
# 创建日志目录和健康数据目录
RUN mkdir -p logs health_data

# 暴露端口
EXPOSE ${PORT}

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 启动服务
CMD ["python", "inference.py"]
