"""
基准数据集下载脚本
自动下载SRBench、UCI等标准数据集
"""

import os
import urllib.request
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: str):
    """下载文件"""
    logger.info(f"Downloading {url} to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    logger.info(f"Downloaded: {output_path}")


def download_uci_datasets(data_dir: str = "data/benchmark/uci"):
    """下载UCI数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    uci_datasets = {
        'concrete': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
        'wine': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'airfoil': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat',
    }
    
    for name, url in uci_datasets.items():
        output_path = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(output_path):
            try:
                download_file(url, output_path)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        else:
            logger.info(f"{name} already exists: {output_path}")


def download_srbench(data_dir: str = "data/benchmark/srbench"):
    """下载SRBench数据集"""
    logger.info("SRBench datasets need to be downloaded manually:")
    logger.info("1. Clone repository: git clone https://github.com/cavalab/srbench")
    logger.info("2. Follow instructions in the repository")
    logger.info("3. Copy datasets to: data/benchmark/srbench/")


def download_feynman(data_dir: str = "data/benchmark/feynman"):
    """下载Feynman数据集"""
    logger.info("Feynman datasets can be downloaded from:")
    logger.info("1. https://space.mit.edu/home/tegmark/aifeynman.html")
    logger.info("2. Or from SRBench repository")
    logger.info("3. Copy datasets to: data/benchmark/feynman/")


def main():
    """主函数"""
    logger.info("Starting benchmark datasets download...")
    
    # 下载UCI数据集
    logger.info("\n=== Downloading UCI datasets ===")
    download_uci_datasets()
    
    # SRBench和Feynman需要手动下载
    logger.info("\n=== SRBench datasets ===")
    download_srbench()
    
    logger.info("\n=== Feynman datasets ===")
    download_feynman()
    
    logger.info("\n=== Download complete! ===")
    logger.info("Please manually download SRBench and Feynman datasets as instructed above.")


if __name__ == "__main__":
    main()

