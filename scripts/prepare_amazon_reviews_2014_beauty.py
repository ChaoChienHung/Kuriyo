"""Compatibility entry point: ``python -m scripts.ensure_amazon_reviews_2014 --category Beauty``."""

from scripts.ensure_amazon_reviews_2014 import prepare


if __name__ == "__main__":
    prepare("Beauty")
