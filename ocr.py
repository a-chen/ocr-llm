#!/usr/bin/env python3

import os
import argparse
import tempfile
import base64
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Any
from dotenv import load_dotenv
from PIL import Image
import io
from pdf2image import convert_from_path
import openai
import google.generativeai as genai

import logging

logging.basicConfig(level=logging.INFO)

def load_environment() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path('.env').resolve()
    logging.info(f"Looking for .env file at: {env_path}")

    if env_path.exists():
        load_dotenv()
        logging.info(".env file loaded successfully.")
    else:
        logging.warning("No .env file found.")

def list_available_providers() -> List[str]:
    """Return a list of providers that have their API keys configured."""
    available = []
    if os.getenv('OPENAI_API_KEY'):
        available.append('openai')
    if os.getenv('GOOGLE_API_KEY'):
        available.append('gemini')
    return available

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def process_image(self, image: bytes) -> str:
        """Process an image and return OCR results."""
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=api_key)

    def process_image(self, image: bytes) -> str:
        encoded_image = base64.b64encode(image).decode('utf-8')
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please perform OCR on this image and return only the extracted text. Correct any potential spelling mistakes"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content

class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def process_image(self, image: bytes) -> str:
        img = Image.open(io.BytesIO(image))
        response = self.model.generate_content([
            "Please perform OCR on this image and return only the extracted text. Correct any potential spelling mistakes",
            img
        ])
        return response.text

class FileHandler:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []

    def process_input(self, input_path: str) -> List[bytes]:
        """Process input file and return list of image bytes."""
        # Expand user path (handles ~) and resolve to absolute path
        path = Path(input_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix.lower() == '.pdf':
            return self._process_pdf(path)
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            return self._process_image(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _process_pdf(self, pdf_path: Path) -> List[bytes]:
        """Convert PDF to images and return as bytes."""
        images = convert_from_path(pdf_path)
        result = []

        for i, image in enumerate(images):
            temp_path = Path(self.temp_dir) / f"page_{i}.jpg"
            image.save(temp_path, 'JPEG')
            self.temp_files.append(temp_path)

            with open(temp_path, 'rb') as f:
                result.append(f.read())

        return result

    def _process_image(self, image_path: Path) -> List[bytes]:
        """Process single image file and return as bytes."""
        with open(image_path, 'rb') as f:
            return [f.read()]

    def cleanup(self):
        """Clean up temporary files and directory."""
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass

        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

class ImageProcessor:
    def prepare_image(self, image_bytes: bytes) -> bytes:
        """Prepare image for OCR processing."""
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert back to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

def get_provider(provider_name: str) -> BaseLLMProvider:
    """Factory function to get the appropriate LLM provider."""
    providers = {
        'openai': OpenAIProvider,
        'gemini': GeminiProvider
    }

    if provider_name not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}")

    return providers[provider_name]()

def main():
    # Load environment variables
    load_environment()

    # Get available providers
    available_providers = list_available_providers()

    if not available_providers:
        raise ValueError(
            "No API keys found. Please set at least one of OPENAI_API_KEY or GOOGLE_API_KEY "
            "in your environment variables or .env file."
        )

    parser = argparse.ArgumentParser(description='Perform OCR on images or PDFs using LLMs')
    parser.add_argument('input_path', help='Path to input file (PDF, JPG, or PNG)', type=str)
    parser.add_argument('--provider', choices=['openai', 'gemini'],
                        default=available_providers[0],  # Use first available provider as default
                        help='LLM provider to use')
    parser.add_argument('--output-format', choices=['md', 'txt'], default='txt',
                        help='Output format')
    parser.add_argument('--output-path', help='Custom output file path')
    args = parser.parse_args()

    # Validate the selected provider's API key
    if args.provider not in available_providers:
        available_str = ", ".join(available_providers)
        raise ValueError(
            f"Selected provider '{args.provider}' is not available. "
            f"Available providers: {available_str}"
        )

    # Initialize components
    file_handler = FileHandler()
    image_processor = ImageProcessor()
    llm_provider = get_provider(args.provider)

    try:
        # Process input file
        input_images = file_handler.process_input(args.input_path)

        # Perform OCR on each image
        results = []
        for img in input_images:
            processed_image = image_processor.prepare_image(img)
            ocr_result = llm_provider.process_image(processed_image)
            results.append(ocr_result)

        # Generate output
        output_text = '\n\n'.join(results)
        if args.output_format == 'markdown':
            output_text = f"# OCR Results\n\n{output_text}"

        # Save output
        if args.output_path:
            output_path = Path(args.output_path).expanduser().resolve()
        else:
            input_filename = Path(args.input_path).expanduser().resolve().stem
            output_path = Path(f"{input_filename}_ocr.{args.output_format}")

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(output_path)

    finally:
        # Cleanup temporary files
        file_handler.cleanup()

if __name__ == '__main__':
    main()
