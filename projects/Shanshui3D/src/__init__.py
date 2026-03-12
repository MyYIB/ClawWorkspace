"""
Shanshui3D - 山水画空间意境重现

基于三远法（高远、深远、平远）的中国山水画 3D 空间意境重现系统。
"""

__version__ = "0.1.0"
__author__ = "小迪 @ OpenClawLab"

from .san_yuan_encoder import SanYuanEncoder
from .reconstruction import ReconstructionModule
from .style_transfer import StyleTransferModule

__all__ = [
    "SanYuanEncoder",
    "ReconstructionModule", 
    "StyleTransferModule",
]
