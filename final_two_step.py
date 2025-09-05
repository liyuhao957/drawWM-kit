#!/usr/bin/env python3
"""
两步法UserID方块增强工具 - 批量处理版
自动检测当前目录的所有JPG和PNG文件
将生成的图层保存到output文件夹中
"""

import cv2
import numpy as np
from scipy import ndimage
import os
import sys
from datetime import datetime
import glob

def step1_super_enhance(image_path):
    """
    步骤1: 生成super_enhanced.png
    复制自test_enhance.py的算法
    """
    print("📌 步骤1: 初步增强...")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 错误：无法读取图片 {image_path}")
        return None
    
    # 1. 锐化增强
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1], 
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # 2. 对比度增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. 组合增强（最强效果）
    kernel2 = np.array([[0,-1,0], 
                        [-1,5,-1], 
                        [0,-1,0]])
    super_sharp = cv2.filter2D(sharpened, -1, kernel2)
    super_enhanced = cv2.addWeighted(super_sharp, 0.8, enhanced, 0.2, 0)
    
    print("  ✅ 初步增强完成")
    return super_enhanced

def step2_generate_ultra_clean(img):
    """
    步骤2A: 生成Ultra Clean版本
    复制自ultra_enhance.py的result_final算法
    """
    print("  生成 Ultra Clean 版本...")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 透明度检测增强
    block_sizes = [4, 8, 16, 32]
    detected_blocks = np.zeros_like(gray, dtype=np.float32)
    
    for size in block_sizes:
        kernel = np.ones((size, size), np.float32) / (size * size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        gray_sq = gray.astype(np.float32) ** 2
        local_mean_sq = cv2.filter2D(gray_sq, -1, kernel)
        local_var = local_mean_sq - local_mean ** 2
        detected_blocks += local_var
    
    detected_blocks = cv2.normalize(detected_blocks, None, 0, 255, cv2.NORM_MINMAX)
    detected_blocks = detected_blocks.astype(np.uint8)
    
    # 2. 频域增强
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    
    mask = np.zeros((rows, cols), dtype=np.float32)
    for r in [10, 20, 30, 40]:
        y, x = np.ogrid[:rows, :cols]
        inner_mask = ((x - ccol)**2 + (y - crow)**2 <= (r+5)**2) & \
                     ((x - ccol)**2 + (y - crow)**2 >= (r-5)**2)
        mask[inner_mask] = 1
    
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_freq = np.fft.ifft2(f_ishift)
    img_freq = np.real(img_freq)
    img_freq = cv2.normalize(img_freq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. 边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharr_x**2 + scharr_y**2)
    
    edges_combined = sobel * 0.3 + np.abs(laplacian) * 0.3 + scharr * 0.4
    edges_combined = cv2.normalize(edges_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4. 纹理增强
    ksize = 21
    sigma = 3
    theta = np.pi/4
    lamda = np.pi/4
    gamma = 0.5
    
    kernel_gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0)
    gabor_response = cv2.filter2D(gray, cv2.CV_32F, kernel_gabor)
    gabor_response = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 5. 超级锐化
    kernel_ultra_sharp = np.array([
        [-1, -2, -1],
        [-2, 13, -2],
        [-1, -2, -1]
    ], dtype=np.float32) / 5
    
    sharpened = cv2.filter2D(img, -1, kernel_ultra_sharp)
    
    # 6. 融合所有增强层
    detected_blocks_3ch = cv2.cvtColor(detected_blocks, cv2.COLOR_GRAY2BGR)
    img_freq_3ch = cv2.cvtColor(img_freq, cv2.COLOR_GRAY2BGR)
    edges_3ch = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)
    gabor_3ch = cv2.cvtColor(gabor_response, cv2.COLOR_GRAY2BGR)
    
    result = sharpened.astype(np.float32)
    result = result + detected_blocks_3ch * 0.3
    result = result + img_freq_3ch * 0.2
    result = result + edges_3ch * 0.4
    result = result + gabor_3ch * 0.1
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 最终增强
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result_lab)
    l = clahe.apply(l)
    result_lab = cv2.merge([l, a, b])
    result_final = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    return result_final, edges_combined  # 同时返回edges版本

def process_image(image_path):
    """
    完整的两步处理流程 - 为每个图片创建独立文件夹
    """
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 为当前图片创建独立的输出文件夹
    output_dir = f"{base_name}_enhanced"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 为 {image_path} 创建文件夹: {output_dir}")
    
    print(f"处理文件: {image_path}")
    
    # 步骤1: 初步增强
    super_enhanced = step1_super_enhance(image_path)
    if super_enhanced is None:
        return False, output_dir
    
    # 步骤2: 深度增强
    print("📌 步骤2: 深度增强...")
    ultra_clean, edges = step2_generate_ultra_clean(super_enhanced)
    
    # 保存最终结果到该图片的专属文件夹
    output_file1 = os.path.join(output_dir, f'{base_name}_ultra_clean.png')
    output_file2 = os.path.join(output_dir, f'{base_name}_edges.png')
    
    cv2.imwrite(output_file1, ultra_clean)
    cv2.imwrite(output_file2, edges)
    
    print(f"  ✅ 已保存: {output_file1}")
    print(f"  ✅ 已保存: {output_file2}")
    
    return True, output_dir

def main():
    """
    主函数 - 批量处理当前目录的所有JPG和PNG文件
    每个原图都会有独立的输出文件夹
    """
    print("="*50)
    print("🎯 两步法UserID方块增强工具 - 批量处理版")
    print("  每个图片将生成独立的输出文件夹")
    print("="*50 + "\n")
    
    # 查找当前目录的所有JPG和PNG文件
    jpg_files = glob.glob("*.jpg") + glob.glob("*.JPG") + glob.glob("*.jpeg") + glob.glob("*.JPEG")
    png_files = glob.glob("*.png") + glob.glob("*.PNG")
    
    all_images = jpg_files + png_files
    
    if not all_images:
        print("❌ 未在当前目录找到任何JPG或PNG文件")
        sys.exit(1)
    
    print(f"📷 找到 {len(all_images)} 个图片文件:")
    for img in all_images:
        print(f"  - {img}")
    print()
    
    # 处理统计
    success_count = 0
    failed_files = []
    created_folders = []
    
    # 批量处理所有图片
    for i, image_path in enumerate(all_images, 1):
        print(f"\n[{i}/{len(all_images)}] " + "="*40)
        try:
            success, output_dir = process_image(image_path)
            if success:
                success_count += 1
                created_folders.append(output_dir)
            else:
                failed_files.append(image_path)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            failed_files.append(image_path)
    
    # 输出处理结果总结
    print("\n" + "="*50)
    print("✨ 批量处理完成！")
    print(f"  成功处理: {success_count}/{len(all_images)} 个文件")
    
    if created_folders:
        print(f"\n📂 创建了 {len(created_folders)} 个输出文件夹:")
        for folder in created_folders:
            print(f"    - {folder}/")
    
    if failed_files:
        print(f"\n⚠️  处理失败的文件:")
        for f in failed_files:
            print(f"    - {f}")
    
    print("="*50)

if __name__ == "__main__":
    main()