 
from paddleocr import PaddleOCR

ocr = PaddleOCR(
        lang="korean",
        use_textline_orientation=True,
        device="gpu",
    )

result = ocr.predict("image.jpg")
res = result[0]
print(dir(res))
print(res)