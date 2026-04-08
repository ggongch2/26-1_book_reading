"""
OCR CER 평가 - 전체 이미지 방식
=================================
전체 이미지를 OCR에 넣고 나온 텍스트를 GT 전체와 CER 비교.
실제 사용 시나리오(책 스캔 → 텍스트 추출)에 가까운 평가 방식.

사용법:
    python eval_cer_fullpage.py --engine paddle --stems test1000_stems.json --max 100
    python eval_cer_fullpage.py --engine easyocr --stems test1000_stems.json --max 100
"""

import io
import json
import time
import argparse
import zipfile
import numpy as np
from pathlib import Path, PurePosixPath
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MIN_BBOX_W = 20
MIN_BBOX_H = 15


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
# 라벨 파싱 - GT 전체 텍스트 합치기
# ───────────────────────────────────────────

def parse_gt_text(data: bytes) -> str:
    """GT bbox 텍스트를 읽기 순서대로 이어붙임"""
    obj = json.loads(data.decode("utf-8"))
    texts = []
    for ann in obj.get("annotations", []):
        text = ann.get("annotation.text", "").strip()
        bbox = ann.get("annotation.bbox")
        if text and bbox and len(bbox) == 4:
            if int(bbox[2]) < MIN_BBOX_W or int(bbox[3]) < MIN_BBOX_H:
                continue
            texts.append(text)
    return " ".join(texts)


# ───────────────────────────────────────────
# 엔진별 전체 이미지 OCR
# ───────────────────────────────────────────

def ocr_fullpage_paddle(ocr, img_np: np.ndarray) -> str:
    result = ocr.predict(img_np)
    texts = []
    for res in result:
        texts.extend(res.get("rec_texts", []))
    return " ".join(texts).strip()

def ocr_fullpage_easyocr(reader, img_np: np.ndarray) -> str:
    results = reader.readtext(img_np, detail=0)
    return " ".join(results).strip()


# ───────────────────────────────────────────
# 평가 루프
# ───────────────────────────────────────────

def eval_loop(ocr_fn, pairs_iter, total, engine_name, save_path):
    all_results = []
    total_ed, total_gt_len = 0, 0

    for i, (name, gt_text, img_np) in enumerate(pairs_iter):
        if not gt_text:
            continue

        t0 = time.time()
        pred_text = ocr_fn(img_np)
        elapsed = time.time() - t0

        cer = calc_cer(gt_text, pred_text)
        ed = edit_distance(gt_text.strip(), pred_text.strip())
        total_ed += ed
        total_gt_len += len(gt_text.strip())

        total_str = str(total) if total else "?"
        print(f"[{i+1}/{total_str}] {name} | CER: {cer:.4f} | {elapsed:.2f}s")

        if cer > 0.4:
            print(f"  GT  (앞50): {gt_text[:50]}")
            print(f"  PRED(앞50): {pred_text[:50]}")

        all_results.append({
            "file": name,
            "cer": round(cer, 4),
            "time_sec": round(elapsed, 3),
            "gt_chars": len(gt_text.strip()),
            "gt": gt_text,
            "pred": pred_text,
        })

    global_cer = total_ed / total_gt_len if total_gt_len > 0 else 0.0
    mean_time = sum(r["time_sec"] for r in all_results) / len(all_results) if all_results else 0.0

    print("\n" + "=" * 60)
    print(f"테스트 완료 : {len(all_results)}개")
    print(f"Global CER  : {global_cer:.4f} ({global_cer*100:.2f}%)")
    print(f"평균 추론   : {mean_time:.3f}s/장")
    print("=" * 60)

    out = {
        "summary": {
            "engine": engine_name,
            "mode": "fullpage",
            "total_images": len(all_results),
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

def iter_from_zips(label_zip, img_zips, stem_set, max_samples):
    print("zip 인덱싱 중...")
    with zipfile.ZipFile(label_zip) as z:
        lbl_map = {PurePosixPath(n).stem: n for n in z.namelist() if n.endswith(".json")}
    img_map = {}
    for zip_path in img_zips:
        with zipfile.ZipFile(zip_path) as z:
            for n in z.namelist():
                if PurePosixPath(n).suffix.lower() in IMAGE_EXTS:
                    img_map[PurePosixPath(n).stem] = (zip_path, n)
    pairs = [(s, lbl_map[s], *img_map[s]) for s in sorted(lbl_map) if s in img_map]
    if stem_set:
        pairs = [p for p in pairs if p[0] in stem_set]
    if max_samples:
        pairs = pairs[:max_samples]
    print(f"평가 대상: {len(pairs)}개")

    zip_handles = {}
    try:
        for stem, lbl_inner, img_zip, img_inner in pairs:
            if label_zip not in zip_handles:
                zip_handles[label_zip] = zipfile.ZipFile(label_zip)
            if img_zip not in zip_handles:
                zip_handles[img_zip] = zipfile.ZipFile(img_zip)
            gt_text = parse_gt_text(zip_handles[label_zip].read(lbl_inner))
            img_np = np.array(Image.open(io.BytesIO(zip_handles[img_zip].read(img_inner))).convert("RGB"))
            yield stem, gt_text, img_np
    finally:
        for z in zip_handles.values():
            z.close()


# ───────────────────────────────────────────
# 엔트리포인트
# ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",    type=str, required=True, choices=["paddle", "easyocr"])
    parser.add_argument("--label_zip", type=str, default="validation/[라벨]validation.zip")
    parser.add_argument("--img_zips",  type=str, nargs="+",
                        default=["validation/[원천]validation.zip"])
    parser.add_argument("--stems",     type=str, default=None)
    parser.add_argument("--max",       type=int, default=None)
    parser.add_argument("--out",       type=str, default=None)
    args = parser.parse_args()

    stem_set = None
    if args.stems:
        with open(args.stems) as f:
            stem_set = set(json.load(f))
        print(f"stem 필터: {len(stem_set)}개")

    out_path = args.out or f"eval_results_{args.engine}_fullpage.json"

    pairs_iter = iter_from_zips(args.label_zip, args.img_zips, stem_set, args.max)

    if args.engine == "paddle":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu",
        )
        eval_loop(lambda img: ocr_fullpage_paddle(ocr, img), pairs_iter, args.max, "PaddleOCR", out_path)

    elif args.engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(["ko"], gpu=True)
        eval_loop(lambda img: ocr_fullpage_easyocr(reader, img), pairs_iter, args.max, "EasyOCR", out_path)
