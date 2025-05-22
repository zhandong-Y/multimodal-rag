import nltk
nltk.data.path = ["/home/amax/nltk_data"]
# resource = nltk.data.find("taggers/averaged_perceptron_tagger")
# print(f"加载成功，路径为: {resource}")
# nltk.data.path.insert(0, "/home/amax/nltk_data")

from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="/home/amax/data3/yzd/multimodal-rag/attention.pdf",
    ocr_strategy="hi_res",  # 强制使用 OCR 模式
    infer_table_structure=True,
)

for el in elements:
    print(el.text, el.metadata.page_number)
