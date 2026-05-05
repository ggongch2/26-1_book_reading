"""
OCR CER 평가 - PPStructureV3 fullpage 방식
==========================================
PPStructureV3로 전체 이미지를 파싱하고 Markdown에서 텍스트를 추출.
GT bbox를 위→아래, 좌→우로 정렬해서 join → 순서 불일치 문제 최소화.

사용법:
    python eval_cer_ppstructure.py --stems ../test1000_stems.json --max 100
    python eval_cer_ppstructure.py --label_zip "../validation/[라벨]validation.zip" \
                                   --img_zips "../validation/[원천]validation.zip"
"""

from __future__ import annotations

import ctypes
import io
import json
import os
import sys
import time
import zipfile
import argparse
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(".paddlex_cache").resolve()))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

libgomp_path = Path(sys.prefix) / "lib" / "libgomp.so.1"
if libgomp_path.exists():
    ctypes.CDLL(str(libgomp_path), mode=ctypes.RTLD_GLOBAL)

from paddleocr import PPStructureV3

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
# GT 파싱 - 위→아래, 좌→우 정렬
# ───────────────────────────────────────────

def parse_gt_text(data: bytes) -> str:
    obj = json.loads(data.decode("utf-8"))
    items = []
    for ann in obj.get("annotations", []):
        text = ann.get("annotation.text", "").strip()
        bbox = ann.get("annotation.bbox")
        if text and bbox and len(bbox) == 4:
            if int(bbox[2]) < MIN_BBOX_W or int(bbox[3]) < MIN_BBOX_H:
                continue
            items.append({"text": text, "bbox": bbox})
    items.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))  # y, x 순 정렬
    return " ".join(item["text"] for item in items)


# ───────────────────────────────────────────
# PPStructureV3 fullpage OCR
# ───────────────────────────────────────────

def ocr_fullpage(pipeline: PPStructureV3, img_np: np.ndarray) -> str:
    img_pil = Image.fromarray(img_np)
    results = list(pipeline.predict(input=img_pil))
    if not results:
        return ""
    if len(results) == 1:
        return results[0].markdown.get("markdown_texts", "").strip()
    markdown_pages = [r.markdown for r in results]
    combined = pipeline.concatenate_markdown_pages(markdown_pages)
    return combined.get("markdown_texts", "").strip()


# ───────────────────────────────────────────
# 이터레이터
# ───────────────────────────────────────────

def iter_from_zips(label_zip: str, img_zips: list[str], stem_set, max_samples):
    print("zip 인덱싱 중...")
    with zipfile.ZipFile(label_zip) as z:
        lbl_map = {PurePosixPath(n).stem: n for n in z.namelist() if n.endswith(".json")}
    img_map: dict[str, tuple[str, str]] = {}
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

    zip_handles: dict[str, zipfile.ZipFile] = {}
    try:
        for stem, lbl_inner, img_zip, img_inner in pairs:
            if label_zip not in zip_handles:
                zip_handles[label_zip] = zipfile.ZipFile(label_zip)
            if img_zip not in zip_handles:
                zip_handles[img_zip] = zipfile.ZipFile(img_zip)
            gt_text = parse_gt_text(zip_handles[label_zip].read(lbl_inner))
            img_np = np.array(
                Image.open(io.BytesIO(zip_handles[img_zip].read(img_inner))).convert("RGB")
            )
            yield stem, gt_text, img_np
    finally:
        for z in zip_handles.values():
            z.close()


# ───────────────────────────────────────────
# 평가 루프
# ───────────────────────────────────────────

def eval_loop(pipeline, pairs_iter, total, save_path):
    all_results = []
    total_ed, total_gt_len = 0, 0

    for i, (name, gt_text, img_np) in enumerate(pairs_iter):
        if not gt_text:
            continue

        t0 = time.time()
        pred_text = ocr_fullpage(pipeline, img_np)
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
            "engine": "PPStructureV3",
            "mode": "fullpage_sorted",
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
# 엔트리포인트
# ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_zip", type=str,
                        default="../validation/[라벨]validation.zip")
    parser.add_argument("--img_zips",  type=str, nargs="+",
                        default=["../validation/[원천]validation.zip"])
    parser.add_argument("--stems",     type=str, default=None)
    parser.add_argument("--max",       type=int, default=None)
    parser.add_argument("--device",    type=str, default="gpu:0")
    parser.add_argument("--out",       type=str,
                        default="eval_results_ppstructure_fullpage.json")
    args = parser.parse_args()

    stem_set = None
    if args.stems:
        with open(args.stems) as f:
            stem_set = set(json.load(f))
        print(f"stem 필터: {len(stem_set)}개")

    print("PPStructureV3 로딩 중...")
    pipeline = PPStructureV3(
        lang="korean",
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
    )
    print("로딩 완료")

    total = args.max
    pairs_iter = iter_from_zips(args.label_zip, args.img_zips, stem_set, args.max)
    eval_loop(pipeline, pairs_iter, total, args.out)
