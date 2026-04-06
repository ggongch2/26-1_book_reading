"""
OCR CER 평가 - TrOCR / bbox 크롭 방식
=========================================
사용법:
    python eval_cer_trocr.py --input ./validation
    python eval_cer_trocr.py --label_zip "[라벨]train.zip" --img_zips "[원천]train1.zip" ...
    python eval_cer_trocr.py --label_zip ... --img_zips ... --max 200
"""

import io
import json
import time
import argparse
import zipfile
import unicodedata
import numpy as np
from pathlib import Path, PurePosixPath
from PIL import Image

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────

MODEL_NAME = "team-lucid/trocr-small-korean"
MIN_BBOX_W = 20
MIN_BBOX_H = 15
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ───────────────────────────────────────────
# CER
# ───────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def calc_cer(gt: str, pred: str) -> float:
    gt, pred = gt.strip(), pred.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    return min(edit_distance(gt, pred) / len(gt), 1.0)


# ───────────────────────────────────────────
# 데이터 수집
# ───────────────────────────────────────────

def gather_pairs_from_dir(root: Path):
    pairs = []
    for json_path in sorted(root.rglob("*.json")):
        img_path = None
        for ext in IMAGE_EXTS:
            c = json_path.with_suffix(ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            img_dir = json_path.parent.parent / "images"
            for ext in IMAGE_EXTS:
                c = img_dir / (json_path.stem + ext)
                if c.exists():
                    img_path = c
                    break
        if img_path:
            pairs.append((str(img_path), str(json_path)))
    return pairs

def gather_pairs_from_zips(label_zip: str, img_zips: list[str]):
    print("라벨 zip 인덱싱 중...")
    label_map = {}
    with zipfile.ZipFile(label_zip) as z:
        for name in z.namelist():
            if name.endswith(".json"):
                label_map[PurePosixPath(name).stem] = name
    print(f"  라벨 {len(label_map)}개")

    print("이미지 zip 인덱싱 중...")
    img_map = {}
    for zip_path in img_zips:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if PurePosixPath(name).suffix.lower() in IMAGE_EXTS:
                    img_map[PurePosixPath(name).stem] = (zip_path, name)
        print(f"  {Path(zip_path).name}: {len(img_map)}개 누적")

    pairs = [
        (stem, label_zip, lbl_inner, *img_map[stem])
        for stem, lbl_inner in sorted(label_map.items())
        if stem in img_map
    ]
    print(f"매칭된 쌍: {len(pairs)}개")
    return pairs


# ───────────────────────────────────────────
# 라벨 파싱
# ───────────────────────────────────────────

def parse_label_bytes(data: bytes) -> list[dict]:
    obj = json.loads(data.decode("utf-8"))
    items = []
    for ann in obj.get("annotations", []):
        text = ann.get("annotation.text", "").strip()
        bbox = ann.get("annotation.bbox", None)
        if text and bbox and len(bbox) == 4:
            if int(bbox[2]) < MIN_BBOX_W or int(bbox[3]) < MIN_BBOX_H:
                continue
            items.append({"text": text, "bbox": bbox})
    return items

def parse_label_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return parse_label_bytes(f.read().encode("utf-8"))


# ───────────────────────────────────────────
# bbox 크롭 → TrOCR
# ───────────────────────────────────────────

def crop_bbox(img: np.ndarray, bbox: list):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def ocr_crop(processor, model, device, crop: np.ndarray) -> str:
    img = Image.fromarray(crop).convert("RGB")
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return unicodedata.normalize("NFC", text).strip()


# ───────────────────────────────────────────
# 평가 루프
# ───────────────────────────────────────────

def eval_loop(processor, model, device, pairs_iter, total, save_path):
    all_results = []
    total_ed, total_gt_len = 0, 0
    skipped = 0

    for i, (name, items, img_np) in enumerate(pairs_iter):
        if not items:
            skipped += 1
            continue

        t0 = time.time()
        pair_results = []
        img_ed, img_gt_len = 0, 0

        for item in items:
            gt = item["text"]
            crop = crop_bbox(img_np, item["bbox"])
            pred = ocr_crop(processor, model, device, crop) if crop is not None else ""

            cer = calc_cer(gt, pred)
            img_ed += edit_distance(gt.strip(), pred.strip())
            img_gt_len += len(gt.strip())
            pair_results.append({"gt": gt, "pred": pred, "cer": round(cer, 4)})

        elapsed = time.time() - t0
        img_cer = img_ed / img_gt_len if img_gt_len > 0 else 0.0
        total_ed += img_ed
        total_gt_len += img_gt_len

        total_str = str(total) if total else "?"
        print(f"[{i+1}/{total_str}] {name} | CER: {img_cer:.4f} | {elapsed:.2f}s | {len(items)}개")

        if img_cer > 0.3:
            worst = sorted(pair_results, key=lambda x: x["cer"], reverse=True)[:3]
            for p in worst:
                print(f"  GT  : {p['gt']}")
                print(f"  PRED: {p['pred'] if p['pred'] else '(미감지)'}")

        all_results.append({
            "file": name,
            "cer": round(img_cer, 4),
            "time_sec": round(elapsed, 3),
            "gt_chars": img_gt_len,
            "pairs": pair_results,
        })

    global_cer = total_ed / total_gt_len if total_gt_len > 0 else 0.0
    mean_time = sum(r["time_sec"] for r in all_results) / len(all_results) if all_results else 0.0

    print("\n" + "=" * 60)
    print(f"테스트 완료 : {len(all_results)}개 (스킵: {skipped}개)")
    print(f"Global CER  : {global_cer:.4f} ({global_cer*100:.2f}%)")
    print(f"평균 추론   : {mean_time:.3f}s/장")
    print("=" * 60)

    out = {
        "summary": {
            "engine": "TrOCR",
            "model": MODEL_NAME,
            "total_images": len(all_results),
            "skipped": skipped,
            "global_cer": round(global_cer, 4),
            "mean_time_sec": round(mean_time, 3),
        },
        "details": all_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {save_path}")


# ───────────────────────────────────────────
# 이터레이터
# ───────────────────────────────────────────

def iter_from_dir(pairs, max_samples):
    if max_samples:
        pairs = pairs[:max_samples]
    for img_p, json_p in pairs:
        items = parse_label_file(json_p)
        img_np = np.array(Image.open(img_p).convert("RGB"))
        yield Path(img_p).name, items, img_np

def iter_from_zips(pairs, max_samples):
    if max_samples:
        pairs = pairs[:max_samples]
    zip_handles = {}
    try:
        for stem, lbl_zip, lbl_inner, img_zip, img_inner in pairs:
            if lbl_zip not in zip_handles:
                zip_handles[lbl_zip] = zipfile.ZipFile(lbl_zip)
            if img_zip not in zip_handles:
                zip_handles[img_zip] = zipfile.ZipFile(img_zip)
            items = parse_label_bytes(zip_handles[lbl_zip].read(lbl_inner))
            img_np = np.array(Image.open(io.BytesIO(zip_handles[img_zip].read(img_inner))).convert("RGB"))
            yield stem, items, img_np
    finally:
        for z in zip_handles.values():
            z.close()


# ───────────────────────────────────────────
# 엔트리포인트
# ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=str, default=None)
    parser.add_argument("--label_zip",  type=str, default=None)
    parser.add_argument("--img_zips",   type=str, nargs="+")
    parser.add_argument("--max",        type=int, default=None)
    parser.add_argument("--out",        type=str, default="eval_results_trocr.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TrOCR 모델 로딩 중: {MODEL_NAME} (device={device})")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("로딩 완료")

    if args.label_zip:
        pairs = gather_pairs_from_zips(args.label_zip, args.img_zips)
        total = min(len(pairs), args.max) if args.max else len(pairs)
        eval_loop(processor, model, device, iter_from_zips(pairs, args.max), total, args.out)
    elif args.input:
        pairs = gather_pairs_from_dir(Path(args.input))
        print(f"수집된 쌍: {len(pairs)}개")
        total = min(len(pairs), args.max) if args.max else len(pairs)
        eval_loop(processor, model, device, iter_from_dir(pairs, args.max), total, args.out)
    else:
        parser.print_help()
