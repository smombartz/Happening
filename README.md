# Event Scraper

A modular Python scraper that discovers event detail pages from listing pages, extracts content using Crawl4AI, and uses local Ollama Mistral to extract normalized event fields.

## Features

- **Event Discovery**: Crawls event listing pages with pagination support
- **Content Extraction**: Uses Crawl4AI to render pages and extract markdown content
- **Image Detection**: Finds best event images (og:image, twitter:image, or main content)
- **LLM Extraction**: Uses local Ollama Mistral to extract structured event fields
- **Deduplication**: Avoids processing duplicate URLs
- **Polite Crawling**: Includes delays between requests
- **JSONL Output**: Saves results in structured JSONL format

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) with Mistral model
- Dependencies listed in `requirements.txt`

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install and run Ollama:
```bash
# Install Ollama (see https://ollama.ai/)
# Then pull the Mistral model
ollama pull mistral
```

3. Make sure Ollama is running:
```bash
ollama serve
```

## Usage

### Basic Usage

```bash
# Discover and scrape events from a listing page
python main.py --url https://example.com/events

# Test mode: discover URLs only (no scraping)
python main.py --url https://example.com/events --test

# Custom output file and pagination limit
python main.py --url https://example.com/events --output my_events.jsonl --max-pages 5

# Enable verbose logging
python main.py --url https://example.com/events --verbose
```

### CLI Options

- `--url URL`: Listing URL to scrape events from (required)
- `--output OUTPUT`: Output JSONL file path (default: events.jsonl)
- `--test`: Test mode - discover and print URLs only, no scraping
- `--max-pages N`: Maximum pages to crawl for discovery (default: 10)
- `--verbose`: Enable debug logging
- `--ollama-url URL`: Ollama API URL (default: http://localhost:11434)
- `--model MODEL`: Ollama model name (default: mistral)

## Output Format

Each line in the output JSONL file contains:

```json
{
  "source_url": "https://example.com/events",
  "detail_url": "https://example.com/events/event-123",
  "content_markdown": "# Event Title\\n\\nEvent description...",
  "extraction": {
    "title": "Event Title",
    "start_date": "2024-06-15",
    "end_date": "2024-06-15",
    "start_time": "19:00",
    "end_time": "22:00",
    "timezone": "America/New_York",
    "location_address": "123 Main St, City, State",
    "image_url": "https://example.com/event-image.jpg",
    "description": "Brief event description...",
    "source_url": "https://example.com/events",
    "detail_url": "https://example.com/events/event-123"
  }
}
```

## Architecture

### Modules

- `main.py` - CLI entrypoint and orchestration
- `link_pattern.py` - Link extraction patterns and heuristics
- `crawler/listing_crawler.py` - Discovers event URLs from listing pages
- `crawler/detail_scraper.py` - Scrapes individual event pages with Crawl4AI
- `extractor/llm_extractor.py` - Extracts structured fields using Ollama
- `utils/url_utils.py` - URL normalization and utilities
- `utils/io_utils.py` - JSONL I/O and logging

### Flow

1. **Discovery**: Crawl listing page(s) to find event detail URLs
2. **Scraping**: Extract markdown content and images from each event page
3. **Extraction**: Use LLM to extract structured event fields
4. **Output**: Save results to JSONL file

## Customization

### Link Patterns

Edit `link_pattern.py` to customize which links are considered event pages:

```python
EVENT_LINK_SUBSTRINGS = ["event", "events", "calendar", ...]
```

### LLM Extraction

The LLM prompt can be customized in `extractor/llm_extractor.py`. The system uses:

- Temperature: 0.0 (deterministic)
- Model: Mistral (configurable)
- JSON validation with Pydantic

## Troubleshooting

### Ollama Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Install/update Mistral model
ollama pull mistral

# Check available models
ollama list
```

### Common Problems

- **No events discovered**: Check link patterns in `link_pattern.py`
- **Scraping fails**: Some sites may block automated requests
- **LLM extraction fails**: Verify Ollama is running and model is available
- **Import errors**: Ensure all dependencies are installed

## Examples

### Test Mode (Discovery Only)

```bash
python main.py --url https://events.example.com --test
```

### Full Pipeline with Custom Settings

```bash
python main.py \
  --url https://events.example.com \
  --output events_2024.jsonl \
  --max-pages 20 \
  --verbose
```