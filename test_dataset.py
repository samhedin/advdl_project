"""
Test suite for dataset
"""
import pytest
from src.dataset import build_dataset


@pytest.fixture
def train_loader():
    loader, _ = build_dataset(data_root="./data", batch_size=64, grayscale=False)
    return loader


@pytest.fixture
def test_loader():
    _, loader = build_dataset(data_root="./data", batch_size=64, grayscale=False)
    return loader


def test_train_images_between_minus_one_and_one(train_loader):
    for img, _ in train_loader:
        assert img.shape == (64, 3, 32, 32)
        assert img.max() <= 1 and img.min() == -1


def test_test_images_between_minus_one_and_one(test_loader):
    for img, _ in test_loader:
        assert img.shape == (64, 3, 32, 32)
        assert img.max() <= 1 and img.min() == -1
