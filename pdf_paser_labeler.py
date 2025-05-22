  # -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

import nltk
nltk.data.path = ["/home/amax/nltk_data"]

from unstructured.partition.pdf import partition_pdf

output_path = '/home/amax/data3/yzd/multimodal-rag/'
file_path = output_path + 'attention.pdf'

import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf

# 处理PDF并获取元素（确保启用坐标提取）
elements = partition_pdf(
    file_path,
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
    # include_page_breaks=True,  # 确保坐标信息包含在metadata中
    coordinates_format="xyxy" 
)

doc = fitz.open(file_path)

scale_x = 612 / 1700 
scale_y = 792 / 2200

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    elements_in_page = [e for e in elements if e.metadata.page_number == page_num + 1]
    
    for elem in elements_in_page:
        coord = elem.metadata.coordinates
        
        if not hasattr(coord, 'points'):
            continue
            
        try:
            points = [
                (float(x), float(y)) 
                for point in coord.points 
                for x, y in [point]  # 安全解包子元组
            ]     
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x0, y0 = min(x_coords)*scale_x, min(y_coords)*scale_y
            x1, y1 = max(x_coords)*scale_x, max(y_coords)*scale_y
            rect = fitz.Rect(
                x0, 
                y0, 
                x1, 
                y1
            )
            page.draw_rect(rect, color=(1, 0, 0), width=2)

            # 标注文字：在框左上角
            text_position = fitz.Point(x0, y0 - 5)  # 稍微往上提一点避免重叠
            page.insert_text(text_position,
                            elem.category,  # 插入元素的类型名
                            fontsize=6, color=(0, 0, 1))  # 蓝色字体，小号字
            
        except Exception as e:
            print(f"绘制错误 @ 元素 {elem.id}: {str(e)}")
            print(f"原始坐标数据: {coord.points}")

doc.save("annotated.pdf")