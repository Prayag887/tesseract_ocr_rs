# syntax=docker/dockerfile:1

########## Stage 1: build OpenCV 5.0.0 + the Rust binary ##########
# OpenCV 5.0 is too new to be in Debian's package repos, so it's built from
# source here to match what this project develops against locally (see
# .cargo/config.toml's OPENCV_* env, which is macOS/homebrew-specific and
# deliberately NOT copied into this image — pkg-config against the OpenCV
# built below is used instead).
# NOTE: trixie (not bookworm) is required. `ort`'s download-binaries feature
# fetches a prebuilt libonnxruntime that references __isoc23_* (glibc >= 2.38)
# and libstdc++'s _M_replace_cold (GCC >= 13); bookworm ships glibc 2.36 /
# GCC 12, so linking fails there with undefined-symbol errors. Trixie is
# glibc 2.41 / GCC 14. CI hits this too, which is why it runs ubuntu-latest.
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

# Only core/imgproc/imgcodecs/geometry/dnn are used by this project (see
# .cargo/config.toml's OPENCV_LINK_LIBS), but no BUILD_LIST restriction is
# applied — OpenCV 5's inter-module dependency graph isn't pinned down here,
# and an unattended Docker build failing on a missed transitive dependency
# is worse than the extra build time of the default module set. Python/Java/
# apps/tests/docs/examples are dropped since nothing here touches them.
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
# models/devanagari_rec_dict.txt is pulled in at compile time via
# include_str! (src/local_ocr.rs), so it must exist before `cargo build`,
# not just in the runtime stage.
COPY models ./models
RUN cargo build --release

########## Stage 2: runtime ##########
FROM debian:trixie-slim AS runtime

# -dev packages (not just the runtime .so) so apt resolves the exact
# correct versioned runtime library from trixie's archive itself, rather
# than this Dockerfile guessing a version-suffixed package name (e.g.
# libavcodec59) that can drift out from under a fixed Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libpng-dev libtiff-dev libwebp-dev libopenexr-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    zlib1g-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib /usr/local/lib
RUN ldconfig

WORKDIR /app
COPY --from=builder /app/target/release/tesseract ./tesseract
COPY models ./models
COPY static ./static

RUN useradd --system --uid 10001 --create-home appuser \
    && mkdir -p scanned_document \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=3000
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["./tesseract"]
