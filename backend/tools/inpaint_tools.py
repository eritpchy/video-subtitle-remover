import multiprocessing
import cv2
import numpy as np

from backend.config import config

def batch_generator(data, max_batch_size):
    """
    根据data大小，生成最大长度不超过max_batch_size的均匀批次数据
    """
    n_samples = len(data)
    # 尝试找到一个比MAX_BATCH_SIZE小的batch_size，以使得所有的批次数量尽量接近
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 处理最后一批可能不足batch_size的情况
    # 如果最后一批少于其他批次，则减小batch_size尝试平衡每批的数量
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 减小批次大小
        num_batches = n_samples // batch_size

    # 生成前num_batches个批次
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 将剩余的数据作为最后一个批次
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]

def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # 为了避免框过小，放大10个像素
            x1 = xmin - config.subtitleAreaDeviationPixel.value
            if x1 < 0:
                x1 = 0
            y1 = ymin - config.subtitleAreaDeviationPixel.value
            if y1 < 0:
                y1 = 0
            x2 = xmax + config.subtitleAreaDeviationPixel.value
            y2 = ymax + config.subtitleAreaDeviationPixel.value
            cv2.rectangle(mask, (x1, y1),
                          (x2, y2), (255, 255, 255), thickness=-1)
    return mask

def get_inpaint_area_by_mask(W, H, h, mask, multiple=1):
        """
        获取字幕去除区域，根据mask来确定需要填补的区域和高度，
        并根据模型要求调整区域大小为指定倍数
        
        Args:
            W: 图像宽度
            H: 图像高度
            h: 检测区域高度
            mask: 遮罩图像
            multiple: 区域尺寸需要满足的倍数，默认为1
        
        Returns:
            调整后的绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
        """
        # 存储绘画区域的列表
        inpaint_area = []
        # 从视频底部的字幕位置开始，假设字幕通常位于底部
        to_H = from_H = H
        # 从底部向上遍历遮罩
        while from_H != 0:
            if to_H - h < 0:
                # 如果下一段会超出顶端，则从顶端开始
                from_H = 0
                to_H = h
            else:
                # 确定段的上边界
                from_H = to_H - h
            # 检查当前段落是否包含遮罩像素
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # 如果不是第一个段落，向下移动以确保没遗漏遮罩区域
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # 确保没有越过底部
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                
                # 使用完整宽度，而不是mask宽度
                left = 0
                right = W
                
                # 调整区域大小为指定倍数
                if multiple > 1:
                    # 计算区域高度
                    height = to_H - from_H
                    # 计算需要调整的高度，使其成为multiple的倍数
                    remainder = height % multiple
                    
                    if remainder != 0:
                        # 需要调整的像素数
                        adjust_pixels = multiple - remainder
                        
                        # 计算区域中心点
                        center_y = (from_H + to_H) / 2
                        
                        # 优先对称扩展
                        if from_H - adjust_pixels/2 >= 0 and to_H + adjust_pixels/2 <= H:
                            # 对称扩展
                            from_H = int(center_y - height/2 - adjust_pixels/2)
                            to_H = int(center_y + height/2 + adjust_pixels/2)
                        # 如果对称扩展会超出边界，尝试对称缩小
                        elif height > multiple:  # 确保缩小后高度至少为multiple
                            # 对称缩小
                            from_H = int(center_y - (height - remainder)/2)
                            to_H = int(center_y + (height - remainder)/2)
                        # 如果无法对称调整，则尝试单边调整
                        else:
                            # 向下扩展
                            if to_H + adjust_pixels <= H:
                                to_H += adjust_pixels
                            # 向上扩展
                            elif from_H - adjust_pixels >= 0:
                                from_H -= adjust_pixels
                            # 如果都不行，则尝试缩小区域
                            elif height > multiple:
                                to_H = from_H + height - remainder
                    
                    # 调整宽度，确保是multiple的倍数
                    width = W  # 使用完整宽度
                    remainder_w = width % multiple
                    
                    if remainder_w != 0:
                        # 需要调整的像素数
                        adjust_pixels_w = multiple - remainder_w
                        
                        # 计算中心点，对称缩小
                        center_x = W / 2
                        left = int(center_x - (width - remainder_w)/2)
                        right = int(center_x + (width - remainder_w)/2)
                
                # 将该段落添加到列表中，格式为(ymin, ymax, xmin, xmax)
                area = (from_H, to_H, left, right)
                if area not in inpaint_area:
                    inpaint_area.append(area)
                else:
                    break
            # 移动到下一个段落
            to_H -= h
        return inpaint_area  # 返回绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]

def expand_frame_ranges(frame_ranges, backward_frame_count, forward_frame_count):
    """
    扩展帧区间列表，向前和向后扩展指定的帧数，并确保区间连续性
    
    Args:
        frame_ranges: 帧区间列表，格式为[(start1, end1), (start2, end2), ...]
        backward_frame_count: 向前扩展的帧数
        forward_frame_count: 向后扩展的帧数
        
    Returns:
        扩展后的帧区间列表，保证连续性
    """
    if not frame_ranges:
        return []
    
    # 按起始帧排序
    sorted_ranges = sorted(frame_ranges)
    expanded_ranges = []
    
    for i, (start, end) in enumerate(sorted_ranges):
        # 向前扩展，但不能小于1
        new_start = max(1, start - backward_frame_count)
        
        # 向后扩展
        new_end = end + forward_frame_count
        
        # 检查是否与下一个区间重叠
        if i < len(sorted_ranges) - 1:
            next_start = sorted_ranges[i + 1][0]
            
            # 如果扩展后的结束帧超过了下一个区间的起始帧
            if new_end >= next_start:
                # 计算中点
                mid_point = (end + next_start) // 2
                
                # 如果区间是连续的(相差1)，则对半平分
                if next_start - end == 1:
                    new_end = end  # 保持原结束帧
                else:
                    # 非连续区间，限制扩展到下一个区间起始帧减去backward_frame_count
                    max_expand = next_start - 1  # 确保不会与下一个区间重叠
                    new_end = min(new_end, max_expand)
        
        # 确保与前一个区间不重叠
        if expanded_ranges:
            prev_end = expanded_ranges[-1][1]
            if new_start <= prev_end:
                # 如果新区间的开始小于等于前一个区间的结束，调整开始位置
                new_start = prev_end + 1
        
        # 确保区间有效（开始不大于结束）
        if new_start <= new_end:
            expanded_ranges.append((new_start, new_end))
        else:
            # 如果调整后区间无效，保留原始区间
            expanded_ranges.append((start, end))
    
    return expanded_ranges

def is_frame_number_in_ab_sections(frame_no, ab_sections):
    """
    检查给定的帧号是否在指定的A/B区间内。

    Args:
        frame_no: 要检查的帧号
        ab_sections: 包含A/B区间的列表，格式为[range(start, end), ...]

    Returns:
        如果帧号在A/B区间内，返回True；否则返回False。
    """
    if ab_sections is None:
        return True
    for section in ab_sections:
        if frame_no in section:
            return True
    return False

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
