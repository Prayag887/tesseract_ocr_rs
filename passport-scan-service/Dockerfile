# syntax=docker/dockerfile:1

########## Stage 1: build OpenCV 5.0.0 + the Rust binary ##########
# Rocky Linux 10 is used so the build matches the RHEL-compatible production
# target rather than Debian.
#
# `ort`'s download-binaries feature may fetch a prebuilt libonnxruntime that
# requires glibc >= 2.38 and GCC/libstdc++ >= 13. Rocky 10 provides glibc 2.39
# and GCC 14, while Rocky/RHEL 9 is too old for those symbols.
FROM rockylinux/rockylinux:10 AS builder

ARG RUST_VERSION=1.90.0
ARG OPENCV_VERSION=5.0.0

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# clang-devel / llvm-devel / openexr-devel are available through CRB.
# zlib-ng-compat-devel provides the traditional zlib ABI on EL10.
#
# FFmpeg development packages are intentionally omitted because videoio is not
# part of this OpenCV build and the service only processes still images.
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

# Install the exact Rust version used by CI/production.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- \
        -y \
        --no-modify-path \
        --profile minimal \
        --default-toolchain "${RUST_VERSION}" \
    && rustc --version \
    && cargo --version

# opencv-rust/bindgen needs libclang.
ENV LIBCLANG_PATH=/usr/lib64

RUN test -e /usr/lib64/libclang.so

RUN git clone \
    --branch "${OPENCV_VERSION}" \
    --depth 1 \
    https://github.com/opencv/opencv.git \
    /opencv-src

# BUILD_LIST contains only the modules required by this service.
#
# geometry depends on flann, so flann is included explicitly.
# ONNX inference goes through `ort`, not opencv::dnn.
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

RUN cargo build --release --locked


########## Stage 2: Rocky/RHEL-compatible runtime ##########
FROM rockylinux/rockylinux:10-minimal AS runtime

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Runtime-only native libraries.
#
# Development headers and FFmpeg are not required in the final image because
# the binary only needs the shared libraries OpenCV was linked against, and
# videoio is excluded from the OpenCV build.
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
        shadow-utils \
    && microdnf clean all

COPY --from=builder /usr/local/lib64 /usr/local/lib64

RUN echo /usr/local/lib64 > /etc/ld.so.conf.d/opencv5.conf \
    && ldconfig

WORKDIR /app

COPY --from=builder /app/target/release/passport-scan-service ./passport-scan-service
COPY models ./models
COPY static ./static

RUN useradd \
        --system \
        --uid 10001 \
        --create-home \
        appuser \
    && mkdir -p /app/scanned_document \
    && chown -R appuser:appuser /app \
    && microdnf -y remove shadow-utils \
    && microdnf clean all

USER appuser

ENV PORT=3003

EXPOSE 3003

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --start-period=15s \
    --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["./passport-scan-service"]