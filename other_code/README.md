# PaddleOCR PP-StructureV3 Inference

이 폴더에 `document.pdf`를 넣고 PP-StructureV3로 OCR detection, layout parsing, 표/수식/도장/차트 인식 결과를 저장하는 예제입니다.

## 환경 생성

```bash
conda env create -f environment.yml
conda activate OCR
```

이미 `OCR` 환경이 있다면:

```bash
conda activate OCR
python -m pip install -r requirements.txt
```

PP-StructureV3 전체 파이프라인에는 `paddlex[ocr]` extra가 필요합니다. 이미 환경을 만든 뒤 위 에러가 나면 다음 명령으로 보강 설치할 수 있습니다.

```bash
python -m pip install -U "paddlex[ocr]>=3.5.0,<3.6.0"
```

## 실행

```bash
python infer_structure.py
```

기본 입력은 `./document.pdf`, 기본 출력 폴더는 `./output`입니다.

자주 쓰는 옵션:

```bash
python infer_structure.py --input document.pdf --output output --device gpu:0
python infer_structure.py --input document.pdf --device cpu --layout-model PP-DocLayout-M --text-det-model PP-OCRv5_mobile_det --text-rec-model PP-OCRv5_mobile_rec --table false --formula false --seal false --chart false
python infer_structure.py --input document.pdf --lang en --layout-model PP-DocLayout-M
python infer_structure.py --input document.pdf --table true
```

결과는 페이지별 JSON, 시각화 이미지, Markdown, HTML, XLSX와 PDF 전체를 합친 `output/document.md`로 저장됩니다.

PaddleOCR/PaddleX 모델 캐시는 기본적으로 이 프로젝트의 `.paddlex_cache/` 아래에 저장됩니다.
기본 실행 장치는 `gpu:0`입니다. CPU로 돌릴 때는 위 CPU 예시처럼 mobile 모델과 낮은 기능 옵션을 쓰는 편이 메모리에 안전합니다.

## 폴더 배치 실행

`data_news` 안의 PDF 전체를 파일별 하위 폴더로 저장하려면:

```bash
conda activate OCR
python batch_structure.py --input-dir data_news --output-dir output_gpu_news --device gpu:0
```

한국어 OCR 모델을 쓰려면:

```bash
python batch_structure.py --input-dir data_news --output-dir output_gpu_news_ko --device gpu:0 --lang korean
```

각 파일 결과는 `output_gpu_news/<PDF 파일명>/`에 저장되고, 진행 로그는 `output_gpu_news/batch_log.csv`에 누적됩니다. 이미 `<PDF 파일명>.md`가 있는 파일은 자동으로 건너뜁니다.
터미널에는 `tqdm` 진행바와 현재 파일, 성공/실패/스킵 개수가 표시됩니다. 각 파일의 자세한 실행 출력은 해당 결과 폴더의 `run.log`에 저장됩니다.

## PDF 페이지를 JPG로 변환

```bash
conda activate OCR
python pdf_to_jpg_pages.py --input data_presentation/OC2_240827_TY3_0095.pdf
```

기본 출력 폴더는 `data_presentation/OC2_240827_TY3_0095_jpg/`입니다.

## 이미지 엣지 디텍션

```bash
conda activate OCR
python edge_detect_image.py --input data_presentation/OC2_240827_TY3_0095_jpg/OC2_240827_TY3_0095_0002.jpg
```

기본 출력 폴더는 입력 이미지 옆의 `<이미지 stem>_edges/`입니다. Canny, adaptive threshold, morphology gradient, overlay 결과를 함께 저장합니다.

## PPStructureV3 CER 평가

전체 이미지를 PPStructureV3로 파싱하고 GT와 CER을 비교합니다.
GT bbox를 위→아래, 좌→우로 정렬해서 join하므로 순서 불일치 문제가 최소화됩니다.

```bash
conda activate OCR

# 기본 실행 (validation zip 사용)
python eval_cer_ppstructure.py

# 테스트셋 필터 + 샘플 수 제한
python eval_cer_ppstructure.py --stems ../test1000_stems.json --max 100

# zip 경로 직접 지정
python eval_cer_ppstructure.py \
    --label_zip "../validation/[라벨]validation.zip" \
    --img_zips "../validation/[원천]validation.zip" \
    --stems ../test1000_stems.json \
    --out eval_results_ppstructure_fullpage.json
```

결과는 `eval_results_ppstructure_fullpage.json`에 저장됩니다.

## 텍스트를 줄이고 도형 contour만 추출

```bash
conda activate OCR
python shape_contours_only.py --input data_presentation/OC2_240827_TY3_0095_jpg/OC2_240827_TY3_0095_0002.jpg
```

기본 출력 폴더는 입력 이미지 옆의 `<이미지 stem>_shape_contours/`입니다. 텍스트가 너무 많이 남으면 `--blur`, `--close-kernel`, `--min-area` 값을 올려보면 됩니다.
