# Weight-only OCI image: RF-DETR Seg Medium person-segmenter for
# Movie-Trimmer (stage C: mask-precision blurring). Re-shipped from the
# fitness-coach weight image (stock COCO-pretrained RF-DETR-Seg-Medium,
# Apache-2.0 weights, Roboflow) so this project stays self-contained.
#
# Build & push:
#   docker build -f rfdetr-seg.weight.Dockerfile \
#     -t ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights:v1 .
#   docker push ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights:v1
FROM scratch
LABEL org.opencontainers.image.title="Movie-Trimmer RF-DETR Seg Medium person-segmenter" \
      org.opencontainers.image.description="Stock COCO-pretrained RF-DETR Seg Medium (Apache-2.0 weights, Roboflow); person class only at runtime. Consumed by the Movie-Trimmer scanner for mask-precision blurring." \
      org.opencontainers.image.source="https://github.com/DarkStar1997/Movie-Trimmer" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY models/person_seg_rfdetr_seg_medium_v1.onnx /models/person_seg_rfdetr_seg_medium_v1.onnx
