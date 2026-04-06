"""
AIHub 공공행정문서 OCR - PaddleOCR CER 테스트 (3.x predict API)
================================================================
사용법:
    # 라벨-이미지 매칭 먼저 확인
    python ocr_test.py --img_dir ./validation/images --label_dir ./validation/labels --check_only

    # 테스트 실행
    python ocr_test.py --img_dir ./validation/images --label_dir ./validation/labels --max 10
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path

from paddleocr import PaddleOCR


# ───────────────────────────────────────────
# 1. CER 계산
# ───────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def calc_cer(gt: str, pred: str) -> float:
    gt, pred = gt.strip(), pred.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(gt, pred) / len(gt)


# ───────────────────────────────────────────
# 2. 라벨 파싱
# ───────────────────────────────────────────

def parse_label(label_path: str) -> list:
    """반환: [{"text": str, "bbox": [x,y,w,h]}, ...]"""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for ann in data.get("annotations", []):
        text = ann.get("annotation.text", "").strip()
        bbox = ann.get("annotation.bbox", None)
        if text and bbox:
            items.append({"text": text, "bbox": bbox})
    return items


# ───────────────────────────────────────────
# 3. IoU 기반 bbox 매칭
# ───────────────────────────────────────────

def xywh_to_xyxy(b):
    return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def parse_ocr_result(result) -> list:
    boxes = []
    for res in result:
        rec_polys = res["rec_polys"]   # list of (4,2) arrays
        rec_texts = res["rec_texts"]
        for pts, text in zip(rec_polys, rec_texts):
            pts = np.array(pts)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            boxes.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "text": text
            })
    return boxes


def match_ocr_to_gt(ocr_boxes, gt_items, iou_thresh=0.3):
    """GT 각 항목에 가장 IoU 높은 OCR 박스 매칭"""
    if not ocr_boxes:
        return [(item["text"], "") for item in gt_items]
    pairs = []
    for gt in gt_items:
        gt_box = xywh_to_xyxy(gt["bbox"])
        best_iou, best_text = 0.0, ""
        for ocr in ocr_boxes:
            s = iou(gt_box, ocr["xyxy"])
            if s > best_iou:
                best_iou, best_text = s, ocr["text"]
        pairs.append((gt["text"], best_text if best_iou >= iou_thresh else ""))
    return pairs


# ───────────────────────────────────────────
# 4. 매칭 확인용
# ───────────────────────────────────────────

def check_mapping(img_dir: Path, label_dir: Path):
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    label_paths = sorted(label_dir.glob("*.json"))
    print(f"이미지 수: {len(img_paths)}")
    print(f"라벨 수:   {len(label_paths)}")
    print(f"이미지 stem 샘플: {[p.stem for p in img_paths[:3]]}")
    print(f"라벨   stem 샘플: {[p.stem for p in label_paths[:3]]}")
    matched = [p for p in img_paths if (label_dir / (p.stem + ".json")).exists()]
    print(f"매칭된 쌍: {len(matched)}개")
    if not matched:
        print("\n→ 파일명 불일치. 라벨 폴더 구조를 확인해봐:")
        print(f"  ls {label_dir} | head -5")


# ───────────────────────────────────────────
# 5. 메인 테스트 루프
# ───────────────────────────────────────────

def run_test(img_dir: str, label_dir: str, max_samples=None, save_results=True):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if max_samples:
        img_paths = img_paths[:max_samples]

    print(f"총 테스트 이미지: {len(img_paths)}개")
    print("=" * 60)

    ocr = PaddleOCR(
        lang="korean",
        #text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu",
    )
    

    all_results = []
    total_ed, total_gt_len = 0, 0
    failed = 0

    for i, img_path in enumerate(img_paths):
        label_path = label_dir / (img_path.stem + ".json")
        if not label_path.exists():
            print(f"[{i+1}] 라벨 없음: {img_path.name} → 스킵")
            failed += 1
            continue

        gt_items = parse_label(str(label_path))
        if not gt_items:
            failed += 1
            continue

        t0 = time.time()
        result = ocr.predict(str(img_path))
        elapsed = time.time() - t0

        ocr_boxes = parse_ocr_result(result)
        pairs = match_ocr_to_gt(ocr_boxes, gt_items)

        img_ed = sum(edit_distance(gt, pred) for gt, pred in pairs)
        img_gt_len = sum(len(gt) for gt, _ in pairs)
        img_cer = img_ed / img_gt_len if img_gt_len > 0 else 0.0

        total_ed += img_ed
        total_gt_len += img_gt_len

        all_results.append({
            "file": img_path.name,
            "cer": round(img_cer, 4),
            "time_sec": round(elapsed, 3),
            "gt_chars": img_gt_len,
            "pairs": [
                {"gt": gt, "pred": pred, "cer": round(calc_cer(gt, pred), 4)}
                for gt, pred in pairs
            ],
        })

        print(f"[{i+1}/{len(img_paths)}] {img_path.name} | CER: {img_cer:.4f} | {elapsed:.2f}s | GT박스: {len(gt_items)}개")

        if img_cer > 0.3:
            worst = sorted(pairs, key=lambda p: calc_cer(*p), reverse=True)[:3]
            for gt, pred in worst:
                print(f"  GT  : {gt}")
                print(f"  PRED: {pred if pred else '(미감지)'}")

    valid = len(all_results)
    global_cer = total_ed / total_gt_len if total_gt_len > 0 else 0.0
    mean_time = sum(r["time_sec"] for r in all_results) / valid if valid > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"테스트 완료 : {valid}개 (스킵: {failed}개)")
    print(f"Global CER  : {global_cer:.4f} ({global_cer*100:.2f}%)")
    print(f"평균 추론   : {mean_time:.3f}s/장")
    print("=" * 60)

    if save_results and valid > 0:
        out = {
            "summary": {
                "engine": "PaddleOCR",
                "lang": "korean",
                "total_images": valid,
                "skipped": failed,
                "global_cer": round(global_cer, 4),
                "mean_time_sec": round(mean_time, 3),
            },
            "details": all_results,
        }
        with open("ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("결과 저장: ocr_results.json")

    return global_cer, all_results


# ───────────────────────────────────────────
# 6. 엔트리포인트
# ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",    type=str, required=True)
    parser.add_argument("--label_dir",  type=str, required=True)
    parser.add_argument("--max",        type=int, default=None)
    parser.add_argument("--no_save",    action="store_true")
    parser.add_argument("--check_only", action="store_true", help="라벨-이미지 매칭 확인만")
    args = parser.parse_args()

    if args.check_only:
        check_mapping(Path(args.img_dir), Path(args.label_dir))
    else:
        run_test(
            img_dir=args.img_dir,
            label_dir=args.label_dir,
            max_samples=args.max,
            save_results=not args.no_save,
        )