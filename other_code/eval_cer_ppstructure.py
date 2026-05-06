"""
OCR CER 평가 - PPStructureV3 fullpage 방식
==========================================
PPStructureV3로 전체 이미지를 파싱하고 Markdown에서 텍스트를 추출.
GT bbox를 위→아래, 좌→우로 정렬해서 join → 순서 불일치 문제 최소화.

데이터 포맷 (JSON 1개 = bbox 1개):
    {
      "source_data_info": {"source_data_name_jpg": "OC2_xxx.jpg"},
      "learning_data_info": {
        "plain_text": "인식할 텍스트",
        "bounding_box": [x, y, w, h]
      }
    }

사용법:
    python eval_cer_ppstructure.py --input ~/OCR/1.데이터/Validation --max 100
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
from collections import defaultdict
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

def parse_gt_from_json(path: Path) -> dict:
    """JSON 1개에서 이미지 이름 + 텍스트 + bbox 추출"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    img_name = obj.get("source_data_info", {}).get("source_data_name_jpg", "")
    info = obj.get("learning_data_info", {})
    text = info.get("plain_text", "").strip()
    bbox = info.get("bounding_box", [])
    return {"img_name": img_name, "text": text, "bbox": bbox}

def gather_pairs(input_dir: Path, max_samples: int | None):
    """디렉토리에서 이미지별로 JSON을 묶어 (img_path, gt_text) 쌍 생성"""
    img_groups: dict[str, list[dict]] = defaultdict(list)
    for json_path in sorted(input_dir.rglob("*.json")):
        try:
            item = parse_gt_from_json(json_path)
        except Exception:
            continue
        if item["img_name"] and item["text"]:
            img_groups[item["img_name"]].append(item)

    pairs = []
    for img_name, items in sorted(img_groups.items()):
        # 이미지 파일 찾기
        img_path = input_dir / img_name
        if not img_path.exists():
            for ext in IMAGE_EXTS:
                cand = input_dir / (Path(img_name).stem + ext)
                if cand.exists():
                    img_path = cand
                    break
        if not img_path.exists():
            continue
        # y, x 순 정렬 후 GT 텍스트 합치기
        items.sort(key=lambda x: (x["bbox"][1] if len(x["bbox"]) >= 2 else 0,
                                   x["bbox"][0] if len(x["bbox"]) >= 1 else 0))
        gt_text = " ".join(it["text"] for it in items)
        pairs.append((img_path, gt_text))

    if max_samples:
        pairs = pairs[:max_samples]
    return pairs

def iter_from_dir(input_dir: Path, max_samples: int | None):
    pairs = gather_pairs(input_dir, max_samples)
    print(f"평가 대상: {len(pairs)}개")
    for img_path, gt_text in pairs:
        img_np = np.array(Image.open(img_path).convert("RGB"))
        yield img_path.name, gt_text, img_np


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
    parser.add_argument("--input",  type=str, required=True,
                        help="JSON + 이미지가 있는 디렉토리 (예: ~/OCR/1.데이터/Validation)")
    parser.add_argument("--max",    type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu:0")
    parser.add_argument("--out",    type=str,
                        default="eval_results_ppstructure_fullpage.json")
    args = parser.parse_args()

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

    pairs_iter = iter_from_dir(Path(args.input), args.max)
    eval_loop(pipeline, pairs_iter, args.max, args.out)
