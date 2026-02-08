"""
Image processor using CLIP for lightweight zero-shot classification.

The model is *lazy-loaded* on the first call to process() and can be
explicitly unloaded afterwards via _unload_model() to reclaim RAM.
"""
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.base import AbstractProcessor, FileType, FileContent
from src.config import get_config


class ImageProcessor(AbstractProcessor):
    """Generates simple descriptions for images using CLIP ViT-B/32."""

    # Pre-defined label set used for zero-shot classification.
    # Intentionally coarse — we trade fine-grained accuracy for speed.
    LABELS = [
        "a photo of a person",
        "a photo of people",
        "a photo of a car",
        "a photo of a vehicle",
        "a photo of a building",
        "a photo of an interior room",
        "a photo of a landscape",
        "a photo of nature",
        "a photo of an animal",
        "a photo of food",
        "a photo of an object",
        "a document or text",
        "a diagram or chart",
        "artwork or illustration",
    ]

    # If the top prediction probability is below this threshold the
    # description falls back to the generic "an image".
    CONFIDENCE_THRESHOLD = 0.3

    def __init__(self):
        self.config = get_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

    # ---------------------------------------------------------- model mgmt

    def _load_model(self):
        """Lazy-load CLIP model and processor onto the chosen device."""
        if self._model is not None:
            return
        print("Loading CLIP model...")
        self._model = CLIPModel.from_pretrained(self.config.vision_model)
        self._processor = CLIPProcessor.from_pretrained(self.config.vision_model)
        self._model.to(self.device)
        self._model.eval()

    def _unload_model(self):
        """Free model memory — call this after a batch is done."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------- interface

    def can_process(self, file_type: FileType) -> bool:
        """Return True only for IMAGE files."""
        return file_type == FileType.IMAGE

    def process(self, file_path: Path) -> FileContent:
        """
        Classify the image and return a short natural-language description.

        Args:
            file_path: Path to the image file

        Returns:
            FileContent with description set (extracted_text is None)
        """
        try:
            self._load_model()

            image = Image.open(file_path).convert('RGB')

            inputs = self._processor(
                text=self.LABELS,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)

            top_prob, top_idx = probs[0].max(dim=0)
            top_prob_val = top_prob.item()

            if top_prob_val >= self.CONFIDENCE_THRESHOLD:
                description = f"{self.LABELS[top_idx.item()]} (confidence: {top_prob_val:.2f})"
            else:
                description = "an image"

            return FileContent(
                file_id=0,
                extracted_text=None,
                description=description,
                is_fully_redacted=False,
                page_count=None
            )

        except Exception as e:
            return FileContent(
                file_id=0,
                extracted_text=None,
                description=f"Failed to process image: {e}",
                is_fully_redacted=False,
                page_count=None
            )

    def __del__(self):
        self._unload_model()
