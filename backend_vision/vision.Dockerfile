FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc \
    && chmod a+r /etc/apt/keyrings/mblt.asc \
    && printf "%s\n" \
        "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse" \
        > /etc/apt/sources.list.d/mobilint.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libopencv-core406t64 \
    libopencv-imgcodecs406t64 \
    libopencv-imgproc406t64 \
    libopencv-videoio406t64 \
    libopencv-highgui406t64 \
    libyaml-cpp0.8 \
    yt-dlp \
    mobilint-qb-runtime \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN mkdir -p /workspace/build /workspace/rc /workspace/mxq

WORKDIR /workspace/build

EXPOSE 8081

CMD ["/workspace/build/src/demo/demo", "--http-port", "8081"]
