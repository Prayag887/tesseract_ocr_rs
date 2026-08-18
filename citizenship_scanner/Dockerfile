# syntax=docker/dockerfile:1

########## Stage 1: build OpenCV 5.0.0 + the Rust binary ##########
# Mirrors the passport service's Dockerfile — kept separate rather than
# shared, matching the repo's standalone-service-per-directory pattern.
# NOTE: trixie (not bookworm) is required — see passport-scan-service/Dockerfile
# for why.
FROM rust:1-trixie AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git pkg-config \
    clang libclang-dev llvm-dev \
    libjpeg62-turbo-dev libpng-dev libtiff-dev libwebp-dev libopenexr-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION=5.0.0
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv-src

RUN cmake -S /opencv-src -B /opencv-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF -DBUILD_opencv_apps=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
    -DWITH_CUDA=OFF -DWITH_CUDNN=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF \
    && cmake --build /opencv-build -j"$(nproc)" \
    && cmake --install /opencv-build \
    && ldconfig

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib64/pkgconfig

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# local_ocr.rs pulls the recognizer's char dict in at compile time via
# include_str!, so models/ must exist in the builder stage too, not just
# the runtime stage's COPY below.
COPY models ./models
RUN cargo build --release

########## Stage 2: runtime ##########
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libpng-dev libtiff-dev libwebp-dev libopenexr-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    zlib1g-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib /usr/local/lib
RUN ldconfig

WORKDIR /app
COPY --from=builder /app/target/release/citizenship-scanner ./citizenship-scanner
COPY models ./models
COPY static ./static

RUN useradd --system --uid 10001 --create-home appuser \
    && mkdir -p scanned_document \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=3004
EXPOSE 3004
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["./citizenship-scanner"]
