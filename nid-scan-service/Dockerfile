# syntax=docker/dockerfile:1

########## Stage 1: build OpenCV 5.0.0 + the Rust binary ##########
# OpenCV 5.0 is not packaged by RHEL/Rocky (EPEL is still on 4.x), so it's
# built from source here to match what this project develops against locally
# (see .cargo/config.toml's OPENCV_* env, which is macOS/homebrew-specific and
# deliberately NOT copied into this image — pkg-config against the OpenCV
# built below is used instead).
#
# NOTE: Rocky 10 (not 9) is required. `ort`'s download-binaries feature
# fetches a prebuilt libonnxruntime that references __isoc23_* (glibc >= 2.38)
# and libstdc++'s _M_replace_cold (GCC >= 13); Rocky 9 ships glibc 2.34 /
# GCC 11, so linking fails there with undefined-symbol errors. Rocky 10 is
# glibc 2.39 / GCC 14. CI hits this too, which is why its job runs inside
# this same rockylinux:10 container rather than on a stock runner.
FROM rockylinux/rockylinux:10 AS builder

# clang-devel / llvm-devel / openexr-devel live in CRB (CodeReady Builder),
# which ships disabled on Rocky. zlib is `zlib-ng-compat-devel` on RHEL 10 —
# `zlib-devel` no longer exists, the zlib-ng build took over the ABI.
RUN dnf -y install dnf-plugins-core \
    && dnf config-manager --set-enabled crb \
    && dnf -y install \
        gcc gcc-c++ make cmake ninja-build git pkgconf-pkg-config \
        clang clang-devel llvm-devel \
        libjpeg-turbo-devel libpng-devel libtiff-devel libwebp-devel openexr-devel \
        zlib-ng-compat-devel openssl-devel \
    && dnf clean all && rm -rf /var/cache/dnf

# The opencv crate generates its bindings with bindgen, which dlopens
# libclang at build time. clang-devel puts it in /usr/lib64 on RHEL.
ENV LIBCLANG_PATH=/usr/lib64

# No rust:1-rockylinux image exists, so the toolchain comes from rustup.
# Pinned rather than floating so a toolchain release can't silently change
# what this image builds.
ARG RUST_VERSION=1.90.0
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain ${RUST_VERSION}

ARG OPENCV_VERSION=5.0.0
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv-src

# BUILD_LIST is restricted to the four modules OPENCV_LINK_LIBS actually
# links (see .cargo/config.toml). Note that dnn is NOT among them: every
# `dnn::` mention left in the Rust sources is in a comment, and ONNX
# inference goes through `ort` (ONNX Runtime), not OpenCV. Dropping dnn also
# drops its protobuf dependency, which is a large part of the build.
#
# An earlier revision deliberately built the full default module set for
# fear of tripping over OpenCV 5's inter-module dependency graph. That fear
# turned out to be unfounded: BUILD_LIST resolves transitive dependencies
# itself — asking for `geometry` pulls in `flann` automatically. Measured on
# a 4-core box, this cuts the compile from 876 ninja targets to 362 (~4 min).
#
# Python/Java/apps/tests/docs/examples are dropped since nothing here uses
# them.
#
# WITH_FFMPEG/WITH_GTK/WITH_V4L are off because RHEL ships no ffmpeg or V4L
# devel packages in its base repos (they need RPM Fusion), and nothing here
# decodes video or opens a window — videoio still builds, just without those
# backends. This is the one substantive difference from the Debian build.
#
# CMAKE_INSTALL_LIBDIR is pinned to lib64 to match where RHEL's own loader
# and pkg-config look; leaving it to CMake's guess is what makes the
# PKG_CONFIG_PATH below go stale.
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
# models/devanagari_rec_dict.txt is pulled in at compile time via
# include_str! (src/local_ocr.rs), so it must exist before `cargo build`,
# not just in the runtime stage.
COPY models ./models
RUN cargo build --release

########## Stage 2: runtime ##########
FROM rockylinux/rockylinux:10-minimal AS runtime

# Unlike Debian, RHEL's runtime packages are unversioned and stable
# (libjpeg-turbo, not libjpeg62-turbo), so the runtime .so packages can be
# named directly here — no need for the Debian file's trick of pulling in
# -devel packages just to let the resolver pick the right SONAME.
# curl is for the HEALTHCHECK below; ca-certificates for outbound TLS.
RUN microdnf -y install \
        libjpeg-turbo libpng libtiff libwebp openexr-libs \
        zlib-ng-compat ca-certificates curl \
    && microdnf clean all && rm -rf /var/cache/yum

COPY --from=builder /usr/local/lib64 /usr/local/lib64
RUN echo /usr/local/lib64 > /etc/ld.so.conf.d/opencv5.conf && ldconfig

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
