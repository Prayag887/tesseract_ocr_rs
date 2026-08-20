# syntax=docker/dockerfile:1

########## Stage 1: build OpenCV 5.0.0 + Rust binary ##########
FROM rockylinux/rockylinux:10 AS builder

ARG RUST_VERSION=1.90.0
ARG OPENCV_VERSION=5.0.0

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Rocky 10 CRB contains clang-devel / llvm-devel / openexr-devel.
# zlib-ng-compat-devel provides the traditional zlib ABI on EL10.
#
# FFmpeg development packages are intentionally not installed. The trimmed
# OpenCV build excludes videoio, so none of libavcodec/libavformat/etc. are
# needed by these scanner services.
RUN dnf -y install dnf-plugins-core \
    && dnf config-manager --set-enabled crb \
    && dnf -y install \
        gcc \
        gcc-c++ \
        make \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
        pkgconf-pkg-config \
        clang \
        clang-devel \
        llvm-devel \
        libjpeg-turbo-devel \
        libpng-devel \
        libtiff-devel \
        libwebp-devel \
        openexr-devel \
        zlib-ng-compat-devel \
        openssl-devel \
        which \
        findutils \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Pin Rust to the same toolchain used by CI/deployment.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- \
        -y \
        --no-modify-path \
        --profile minimal \
        --default-toolchain "${RUST_VERSION}" \
    && rustc --version \
    && cargo --version

# Bindgen used by opencv-rust needs libclang.
ENV LIBCLANG_PATH=/usr/lib64

RUN test -e /usr/lib64/libclang.so

RUN git clone \
    --branch "${OPENCV_VERSION}" \
    --depth 1 \
    https://github.com/opencv/opencv.git \
    /opencv-src

# BUILD_LIST is deliberately limited to the modules this service uses.
#
# geometry depends on flann, so flann is explicitly included.
# ONNX inference goes through `ort`, not opencv::dnn.
#
# lib64 is explicitly selected to match the normal RHEL/Rocky library layout.
RUN cmake \
        -S /opencv-src \
        -B /opencv-build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib64 \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_LIST=core,imgproc,imgcodecs,geometry,flann \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_CUDNN=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=OFF \
        -DWITH_ITT=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GTK=OFF \
        -DWITH_V4L=OFF \
    && cmake --build /opencv-build -j"$(nproc)" \
    && cmake --install /opencv-build \
    && echo /usr/local/lib64 > /etc/ld.so.conf.d/opencv5.conf \
    && ldconfig

ENV PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

# local_ocr.rs pulls the recognizer char dictionary in at compile time via
# include_str!, so models must also exist in the builder.
COPY models ./models

RUN cargo build --release --locked


########## Stage 2: Rocky/RHEL-compatible runtime ##########
FROM rockylinux/rockylinux:10-minimal AS runtime

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Runtime only needs the native libraries OpenCV was linked against plus
# curl for HEALTHCHECK.
#
# Use runtime libraries, not -devel packages.
RUN microdnf -y install \
        libjpeg-turbo \
        libpng \
        libtiff \
        libwebp \
        openexr-libs \
        zlib-ng-compat \
        libstdc++ \
        libgcc \
        openssl-libs \
        ca-certificates \
        curl \
    && microdnf clean all

COPY --from=builder /usr/local/lib64 /usr/local/lib64

RUN echo /usr/local/lib64 > /etc/ld.so.conf.d/opencv5.conf \
    && ldconfig

WORKDIR /app

COPY --from=builder /app/target/release/citizenship-scanner ./citizenship-scanner
COPY models ./models
COPY static ./static

# Rocky's minimal image may not provide Debian's useradd semantics/packages
# by default, so shadow-utils is installed temporarily to create the user.
USER root

RUN microdnf -y install shadow-utils \
    && useradd \
        --system \
        --uid 10001 \
        --create-home \
        appuser \
    && mkdir -p /app/scanned_document \
    && chown -R appuser:appuser /app \
    && microdnf -y remove shadow-utils \
    && microdnf clean all

USER appuser

ENV PORT=3004

EXPOSE 3004

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --start-period=15s \
    --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["./citizenship-scanner"]