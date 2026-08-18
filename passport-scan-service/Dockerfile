# syntax=docker/dockerfile:1

########## Stage 1: build OpenCV 5.0.0 + the Rust binary ##########
# Mirrors the NID service's Dockerfile — kept separate rather than shared,
# matching this service's standalone-repo-style isolation. See that file for
# why OpenCV is built from source, why CRB has to be enabled, and why
# .cargo/config.toml (macOS/homebrew-specific) is not copied in.
#
# NOTE: Rocky 10 (not 9) is required. `ort`'s download-binaries feature
# fetches a prebuilt libonnxruntime that references __isoc23_* (glibc >= 2.38)
# and libstdc++'s _M_replace_cold (GCC >= 13); Rocky 9 ships glibc 2.34 /
# GCC 11, so linking fails there with undefined-symbol errors. Rocky 10 is
# glibc 2.39 / GCC 14. CI hits this too, which is why its job runs inside
# this same rockylinux:10 container rather than on a stock runner.
FROM rockylinux/rockylinux:10 AS builder

RUN dnf -y install dnf-plugins-core \
    && dnf config-manager --set-enabled crb \
    && dnf -y install \
        gcc gcc-c++ make cmake ninja-build git pkgconf-pkg-config \
        clang clang-devel llvm-devel \
        libjpeg-turbo-devel libpng-devel libtiff-devel libwebp-devel openexr-devel \
        zlib-ng-compat-devel openssl-devel \
    && dnf clean all && rm -rf /var/cache/dnf

ENV LIBCLANG_PATH=/usr/lib64

ARG RUST_VERSION=1.90.0
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_VERSION}

ARG OPENCV_VERSION=5.0.0
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv-src

RUN cmake -S /opencv-src -B /opencv-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs,geometry \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_JAVA=OFF -DBUILD_opencv_apps=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
    -DWITH_CUDA=OFF -DWITH_CUDNN=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GTK=OFF -DWITH_V4L=OFF \
    && cmake --build /opencv-build -j"$(nproc)" \
    && cmake --install /opencv-build \
    && echo /usr/local/lib64 > /etc/ld.so.conf.d/opencv5.conf \
    && ldconfig

ENV PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

########## Stage 2: runtime ##########
FROM rockylinux/rockylinux:10-minimal AS runtime

RUN microdnf -y install \
        libjpeg-turbo libpng libtiff libwebp openexr-libs \
        zlib-ng-compat ca-certificates curl \
    && microdnf clean all && rm -rf /var/cache/yum

COPY --from=builder /usr/local/lib64 /usr/local/lib64
RUN echo /usr/local/lib64 > /etc/ld.so.conf.d/opencv5.conf && ldconfig

WORKDIR /app
COPY --from=builder /app/target/release/passport-scan-service ./passport-scan-service
COPY models ./models
COPY static ./static

RUN useradd --system --uid 10001 --create-home appuser \
    && mkdir -p scanned_document \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=3003
EXPOSE 3003
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["./passport-scan-service"]
