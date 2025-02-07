# LLM-based OCR Project

This Python script performs OCR (Optical Character Recognition) on images and PDFs using Large Language Models (LLMs). It currently supports OpenAI's GPT-4 Vision and Google's Gemini Pro Vision models.

## Prerequisites

- Python 3.8 or higher
- API keys for:
    - OpenAI (Get it from: https://platform.openai.com/account/api-keys)
    - Google AI (Get it from: https://makersuite.google.com/app/apikey)

## Installation

1. Clone this repository or download the script:
```bash
git clone <repository-url>
# or just download the ocr.py file and requirements.txt
```

2. Create and activate a virtual environment (recommended):
# On Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```
# On macOS/Linux
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Additional requirement for PDF processing:
   - On Windows: Download and install poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
   - On macOS: `brew install poppler`
   - On Linux: `sudo apt-get install poppler-utils`

## Configuration

1. Create a `.env` file in the same directory as the script:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Alternatively, you can set these as environment variables in your system.

## Usage

The script can be run from the command line with various options:

```bash
python ocr.py input_file [--provider PROVIDER] [--output-format FORMAT] [--output-path PATH]
```

### Arguments

- `input_file`: Path to the input file (required)
  - Supported formats: PDF, JPG, PNG
- `--provider`: LLM provider to use (optional)
  - Choices: 'openai' (default) or 'gemini'
- `--output-format`: Output format (optional)
  - Choices: 'txt' (default) or 'md'
- `--output-path`: Custom output file path (optional)
  - Default: `{input_filename}_ocr.{format}`

### Examples

1. Basic usage with default options (OpenAI provider, text output):
```bash
python ocr.py document.pdf
```

2. Using Google Gemini with markdown output:
```bash
python ocr.py image.jpg --provider gemini --output-format md 
```

3. Specifying a custom output path:
```bash
python ocr.py scan.png --output-path result.txt
```

## Output

- The script will process the input file and save the OCR results to the specified output file
- Only the path to the output file will be printed to the console
- For PDFs with multiple pages, the text from each page will be separated by blank lines

## Error Handling

The script will display appropriate error messages for:
- Missing API keys
- Invalid input file paths
- Unsupported file types
- API errors from providers

## Limitations

- PDF processing requires poppler to be installed on the system
- The script processes one file at a time
- API rate limits and token limits apply based on your OpenAI/Google API plan

## Extending

To add support for a new LLM provider:
1. Create a new class that inherits from `BaseLLMProvider`
2. Implement the `process_image` method
3. Add the provider to the `providers` dictionary in the `get_provider()` function
