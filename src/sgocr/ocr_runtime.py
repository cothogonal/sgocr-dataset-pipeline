from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .consensus import OCRVote, geometric_mean
from .semantic_dev40_tuning import load_semantic_dev40_tuning


PADDLE_DISABLE_SOURCE_CHECK_ENV = "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"


@dataclass(frozen=True)
class DetectionRow:
    image_id: str
    node_id: str
    image_path: str
    image_width: int
    image_height: int
    bbox_xyxy: list[float]
    polygon: list[float]
    detection_confidence: float


class CraftDetector:
    def __init__(
        self,
        *,
        text_threshold: float = 0.5,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        long_size: int = 1280,
        device: str = "cpu",
        refiner: bool = True,
    ) -> None:
        import cv2
        import craft_text_detector.craft_utils as craft_utils
        import craft_text_detector.image_utils as image_utils
        import torchvision.models.vgg as tv_vgg
        from craft_text_detector import Craft
        from torchvision.models import VGG16_BN_Weights

        if not hasattr(tv_vgg, "model_urls"):
            tv_vgg.model_urls = {"vgg16_bn": VGG16_BN_Weights.IMAGENET1K_V1.url}

        self.cv2 = cv2
        self.craft_utils = craft_utils
        self.image_utils = image_utils
        self.use_cuda = device.startswith("cuda")
        self.detector = Craft(
            output_dir=None,
            rectify=False,
            export_extra=False,
            text_threshold=float(text_threshold),
            link_threshold=float(link_threshold),
            low_text=float(low_text),
            cuda=bool(self.use_cuda),
            long_size=int(long_size),
            refiner=bool(refiner),
            crop_type="box",
        )
        self.text_threshold = float(text_threshold)
        self.link_threshold = float(link_threshold)
        self.low_text = float(low_text)
        self.long_size = int(long_size)

    def detect(self, image: np.ndarray, *, max_detections: int = 72) -> tuple[list[list[float]], list[float], float]:
        started = time.perf_counter()
        img = self.image_utils.read_image(image)
        img_resized, target_ratio, _ = self.image_utils.resize_aspect_ratio(img, self.long_size, interpolation=self.cv2.INTER_LINEAR)
        ratio_h = ratio_w = 1.0 / float(target_ratio)
        x = self.image_utils.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        if self.use_cuda:
            x = x.cuda(non_blocking=True)

        with torch.no_grad():
            y, feature = self.detector.craft_net(x)
            score_link = y[0, :, :, 1].detach().float().cpu().numpy()
            if self.detector.refine_net is not None:
                y_refiner = self.detector.refine_net(y, feature)
                score_link = y_refiner[0, :, :, 0].detach().float().cpu().numpy()
        score_text = y[0, :, :, 0].detach().float().cpu().numpy()

        boxes, labels, mapper = self.craft_utils.getDetBoxes_core(
            score_text,
            score_link,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
        )
        boxes = self.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

        height, width = int(img.shape[0]), int(img.shape[1])
        detections: list[list[float]] = []
        confidences: list[float] = []
        for box, component_id in zip(boxes, mapper):
            if box is None:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            x1 = max(0.0, min(xs))
            y1 = max(0.0, min(ys))
            x2 = min(float(width), max(xs))
            y2 = min(float(height), max(ys))
            if x2 <= x1 or y2 <= y1:
                continue
            mask = labels == int(component_id)
            confidence = float(np.max(score_text[mask])) if np.any(mask) else 0.0
            detections.append([x1, y1, x2, y2])
            confidences.append(confidence)

        ranked = sorted(zip(detections, confidences), key=lambda item: (-item[1], item[0][1], item[0][0]))[:max_detections]
        return [item[0] for item in ranked], [item[1] for item in ranked], time.perf_counter() - started

    def close(self) -> None:
        self.detector.unload_craftnet_model()
        if getattr(self.detector, "refine_net", None) is not None:
            self.detector.unload_refinenet_model()


class PaddleOCRDetector:
    def __init__(
        self,
        *,
        model_name: str = "PP-OCRv5_server_det",
        limit_side_len: int = 1280,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.5,
        device: str = "cpu",
    ) -> None:
        os.environ.setdefault(PADDLE_DISABLE_SOURCE_CHECK_ENV, "True")
        import paddle
        from paddleocr import TextDetection

        self.device = "gpu:0" if device.startswith("cuda") and paddle.device.is_compiled_with_cuda() else "cpu"
        self.model_name = model_name
        self.detector = TextDetection(
            model_name=model_name,
            device=self.device,
            limit_side_len=int(limit_side_len),
            thresh=float(thresh),
            box_thresh=float(box_thresh),
            unclip_ratio=float(unclip_ratio),
        )

    def detect(self, image: np.ndarray, *, max_detections: int = 72) -> tuple[list[list[float]], list[float], float, list[list[float]]]:
        started = time.perf_counter()
        results = list(self.detector.predict([image]))
        if not results:
            return [], [], time.perf_counter() - started, []
        result = results[0]
        polygons_array = result.get("dt_polys") if hasattr(result, "get") else None
        scores = list(result.get("dt_scores") or []) if hasattr(result, "get") else []
        polygons: list[list[float]] = []
        boxes: list[list[float]] = []
        if polygons_array is None:
            return [], [], time.perf_counter() - started, []
        for poly in np.asarray(polygons_array, dtype=np.float32):
            if poly.shape[0] < 4:
                continue
            flat = [round(float(value), 2) for point in poly.tolist() for value in point[:2]]
            xs = flat[0::2]
            ys = flat[1::2]
            x1 = max(0.0, min(xs))
            y1 = max(0.0, min(ys))
            x2 = max(xs)
            y2 = max(ys)
            if x2 <= x1 or y2 <= y1:
                continue
            polygons.append(flat[:8])
            boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
        ranked = sorted(
            zip(boxes, scores[: len(boxes)], polygons),
            key=lambda item: (-float(item[1]), item[0][1], item[0][0]),
        )[:max_detections]
        return [item[0] for item in ranked], [float(item[1]) for item in ranked], time.perf_counter() - started, [item[2] for item in ranked]

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()


def bbox_to_polygon(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def crop_with_padding(image: Image.Image, box: list[float], *, pad_ratio: float = 0.04) -> Image.Image:
    if pad_ratio == 0.04:
        pad_ratio = float(load_semantic_dev40_tuning().ocr_crop_pad_ratio)
    x1, y1, x2, y2 = [float(value) for value in box]
    pad = max((x2 - x1), (y2 - y1)) * pad_ratio
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(image.width, int(math.ceil(x2 + pad)))
    bottom = min(image.height, int(math.ceil(y2 + pad)))
    return image.crop((left, top, right, bottom)).convert("RGB")


class PARSeqRecognizer:
    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self.model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).to(device).eval()
        self.transform = Compose(
            [
                Resize(tuple(int(value) for value in self.model.hparams.img_size), InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(0.5, 0.5),
            ]
        )

    def recognize(self, crops: list[Image.Image], *, batch_size: int = 24) -> list[OCRVote]:
        return _pick_best_rotations(
            crops,
            infer_variants=lambda variant_images: self._infer_once(variant_images, batch_size=batch_size),
        )

    def _infer_once(self, crops: list[Image.Image], *, batch_size: int) -> list[OCRVote]:
        out: list[OCRVote] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            tensors = torch.stack([self.transform(crop) for crop in batch]).to(self.device)
            with torch.inference_mode():
                probs = self.model(tensors).softmax(-1)
            preds, confidences = self.model.tokenizer.decode(probs)
            for pred, confidence_tensor in zip(preds, confidences):
                confidence = geometric_mean(confidence_tensor.detach().float().cpu().tolist())
                out.append(OCRVote(model_name="parseq", text=str(pred).strip(), confidence=confidence))
        return out


class PaddleOCRRecognizer:
    def __init__(self, model_name: str = "PP-OCRv5_server_rec", *, device: str = "cpu") -> None:
        os.environ.setdefault(PADDLE_DISABLE_SOURCE_CHECK_ENV, "True")
        import paddle
        from paddleocr import TextRecognition

        self.device = "gpu:0" if device.startswith("cuda") and paddle.device.is_compiled_with_cuda() else "cpu"
        self.model_name = model_name
        self.short_name = model_name
        self.recognizer = TextRecognition(model_name=model_name, device=self.device)

    def recognize(self, crops: list[Image.Image], *, batch_size: int = 24) -> list[OCRVote]:
        out: list[OCRVote] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            arrays = [np.asarray(crop.convert("RGB")) for crop in batch]
            results = list(self.recognizer.predict(arrays))
            for row in results:
                text = str(row.get("rec_text") or "").strip()
                confidence = float(row.get("rec_score") or 0.0)
                out.append(OCRVote(model_name=self.short_name, text=text, confidence=confidence))
        return out


class TrOCRRecognizer:
    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        self.device = device
        self.model_name = model_name
        self.short_name = model_name.rsplit("/", 1)[-1].replace("-printed", "")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()
        self.special_ids = set(self.processor.tokenizer.all_special_ids)

    def recognize(self, crops: list[Image.Image], *, batch_size: int = 8) -> list[OCRVote]:
        return _pick_best_rotations(
            crops,
            infer_variants=lambda variant_images: self._infer_once(variant_images, batch_size=batch_size),
        )

    def _infer_once(self, crops: list[Image.Image], *, batch_size: int) -> list[OCRVote]:
        out: list[OCRVote] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    inputs.pixel_values,
                    max_new_tokens=32,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            texts = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)
            for row_index, text in enumerate(texts):
                token_confidences = []
                chosen = generated.sequences[row_index].tolist()[1:]
                for step_index, token_id in enumerate(chosen):
                    if step_index >= len(generated.scores):
                        break
                    if token_id in self.special_ids:
                        continue
                    probs = generated.scores[step_index][row_index].softmax(dim=-1)
                    token_confidences.append(float(probs[token_id]))
                confidence = geometric_mean(token_confidences)
                out.append(OCRVote(model_name=self.short_name, text=str(text).strip(), confidence=confidence))
        return out


def _pick_best_rotations(
    crops: list[Image.Image],
    *,
    infer_variants: Any,
) -> list[OCRVote]:
    variants: list[tuple[int, int, Image.Image]] = []
    for crop_index, crop in enumerate(crops):
        variants.append((crop_index, 0, crop))
        if crop.height > crop.width * 1.25:
            variants.append((crop_index, 90, crop.rotate(90, expand=True)))
            variants.append((crop_index, 270, crop.rotate(270, expand=True)))

    variant_votes = infer_variants([image for _, _, image in variants])
    best_by_crop: dict[int, OCRVote] = {}
    for (crop_index, rotation, _), vote in zip(variants, variant_votes):
        rotated_vote = OCRVote(
            model_name=vote.model_name,
            text=vote.text,
            confidence=vote.confidence,
            rotation=rotation,
        )
        prev = best_by_crop.get(crop_index)
        if prev is None or rotated_vote.confidence > prev.confidence:
            best_by_crop[crop_index] = rotated_vote
    return [best_by_crop[index] for index in range(len(crops))]
