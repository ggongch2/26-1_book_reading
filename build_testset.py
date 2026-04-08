"""
테스트셋 선별 스크립트
=======================
AI Hub 공공행정문서 validation zip에서 테이블 밀집도가 낮은 문서를 선별합니다.

사용법:
    python build_testset.py --n 1000 --max_score 7 --out test1000_stems.json

테이블 score 기준:
    한 y행(±15px)에 bbox가 4개 이상인 행의 수
    score <= 7 → 1,380개 (결재란 정도는 있지만 테이블 없음 수준)
"""

import json
import random
import argparse
import zipfile
from pathlib import PurePosixPath


def table_score(annotations: list) -> int:
    """테이블 밀집도 점수: 같은 y행에 bbox 4개 이상인 행 수"""
    ys = []
    for ann in annotations:
        bbox = ann.get("annotation.bbox")
        if bbox and len(bbox) == 4:
            ys.append(bbox[1] + bbox[3] / 2)
    ys.sort()
    dense_rows = 0
    i = 0
    while i < len(ys):
        group = [ys[i]]
        j = i + 1
        while j < len(ys) and ys[j] - ys[i] <= 15:
            group.append(ys[j])
            j += 1
        if len(group) >= 4:
            dense_rows += 1
        i = j
    return dense_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_zip",  type=str,
                        default="validation/[라벨]validation.zip")
    parser.add_argument("--max_score",  type=int, default=7,
                        help="테이블 score 상한 (기본 7)")
    parser.add_argument("--n",          type=int, default=1000,
                        help="선별 개수")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        type=str, default="test1000_stems.json")
    args = parser.parse_args()

    print(f"라벨 zip 인덱싱: {args.label_zip}")
    candidates = []
    with zipfile.ZipFile(args.label_zip) as z:
        jsons = [n for n in z.namelist() if n.endswith(".json")]
        print(f"전체 {len(jsons)}개 스캔 중...")
        for name in jsons:
            import json as _json
            data = _json.loads(z.read(name).decode("utf-8"))
            sc = table_score(data.get("annotations", []))
            if sc <= args.max_score:
                candidates.append(PurePosixPath(name).stem)

    print(f"score <= {args.max_score}: {len(candidates)}개")

    if len(candidates) < args.n:
        print(f"Warning: 후보({len(candidates)})가 요청({args.n})보다 적음 → 전체 사용")
        selected = sorted(candidates)
    else:
        random.seed(args.seed)
        selected = sorted(random.sample(candidates, args.n))

    with open(args.out, "w", encoding="utf-8") as f:
        _json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"{len(selected)}개 저장 → {args.out}")
