# Event Scraper

A modular Python scraper with dual extraction modes: AI-powered extraction using Crawl4AI + Ollama, or fast rule-based extraction using Scrapy.

## Features

### Core Features
- **Event Discovery**: Crawls event listing pages with pagination support
- **Dual Extraction Modes**: Choose between AI intelligence or speed
- **Event Tagging**: AI mode categorizes events by type and topics
- **Image Detection**: Finds best event images (og:image, twitter:image, or main content)
- **Deduplication**: Avoids processing duplicate URLs
- **Polite Crawling**: Includes delays between requests
- **JSONL Output**: Saves results in structured JSONL format

### LLM Mode (AI-Powered)
- **Content Extraction**: Uses Crawl4AI to render pages and extract markdown content
- **LLM Extraction**: Uses local Ollama Mistral to extract structured event fields
- **Smart Tagging**: Automatically categorizes events by type (Class, Show, etc.) and topics
- **Date Normalization**: Converts various date formats to ISO standard
- **Context Understanding**: AI interprets event content for accurate field extraction

### Scrapy Mode (Rule-Based)
- **Fast Extraction**: Uses CSS selectors and XPath for rapid data extraction
- **No AI Dependency**: Works without Ollama or any LLM
- **Structured Data Parsing**: Handles JSON-LD and microdata automatically
- **Lightweight**: Minimal resource usage for large-scale scraping

## Requirements

### Core Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Mode-Specific Requirements
- **LLM Mode**: [Ollama](https://ollama.ai/) with Mistral model
- **Scrapy Mode**: No additional requirements (Scrapy included in requirements.txt)

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. LLM Mode Setup (Optional)
Only required if you plan to use `--mode llm` (default):

```bash
# Install Ollama (see https://ollama.ai/)
# Then pull the Mistral model
ollama pull mistral

# Make sure Ollama is running
ollama serve
```

### 3. Quick Start
```bash
# Fast Scrapy mode (no AI setup needed)
python main.py --url https://example.com/events --mode scrapy

# AI-powered LLM mode (requires Ollama)  
python main.py --url https://example.com/events --mode llm
```

## Usage

### Mode Selection

```bash
# LLM Mode (default) - AI extraction with event types and topics
python main.py --url https://example.com/events --mode llm

# Scrapy Mode - Fast rule-based extraction (no AI)
python main.py --url https://example.com/events --mode scrapy
```

### Basic Examples

```bash
# Discover and scrape with AI tagging (requires Ollama)
python main.py --url https://example.com/events

# Fast extraction without AI dependency
python main.py --url https://example.com/events --mode scrapy

# Test mode: discover URLs only (no scraping)
python main.py --url https://example.com/events --test

# Custom settings for both modes
python main.py --url https://example.com/events --mode scrapy --output my_events.jsonl --max-pages 5 --verbose
```

### Advanced Examples

```bash
# LLM mode with custom Ollama settings
python main.py \
  --url https://example.com/events \
  --mode llm \
  --ollama-url http://localhost:11434 \
  --model mistral \
  --max-events 50

# Scrapy mode for large-scale scraping
python main.py \
  --url https://example.com/events \
  --mode scrapy \
  --max-pages 20 \
  --max-events 1000 \
  --output bulk_events.jsonl
```

### CLI Options

#### Core Options
- `--url URL`: Listing URL to scrape events from (required)
- `--mode {llm,scrapy}`: Extraction mode (default: llm)
- `--output OUTPUT`: Output JSONL file path (default: events.jsonl)
- `--test`: Test mode - discover and print URLs only, no extraction
- `--max-pages N`: Maximum pages to crawl for discovery (default: 10)
- `--max-events N`: Maximum events to process (default: unlimited)
- `--verbose`: Enable debug logging

#### LLM Mode Only
- `--ollama-url URL`: Ollama API URL (default: http://localhost:11434)
- `--model MODEL`: Ollama model name (default: mistral)

## Output Format

Both modes produce consistent JSONL output. Each line contains:

### Common Structure
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
    "event_type": "Show",
    "topics": ["Music", "Community"],
    "source_url": "https://example.com/events",
    "detail_url": "https://example.com/events/event-123"
  }
}
```

### Mode Differences

| Field | LLM Mode | Scrapy Mode |
|-------|----------|-------------|
| `content_markdown` | Full scraped content | Empty string `""` |
| `event_type` | AI-categorized (e.g., "Show", "Class") | `null` |
| `topics` | AI-tagged array (e.g., ["Music", "Art"]) | Empty array `[]` |
| `timezone` | AI-inferred when possible | `null` |
| Other fields | AI-extracted and normalized | Rule-based extraction |

## Architecture

### Modules

- `main.py` - CLI entrypoint and mode orchestration
- `link_pattern.py` - Link extraction patterns and heuristics
- `crawler/listing_crawler.py` - Discovers event URLs from listing pages (LLM mode)
- `crawler/detail_scraper.py` - Scrapes individual event pages with Crawl4AI (LLM mode)
- `crawler/scrapy_scraper.py` - Scrapy-based extraction pipeline (Scrapy mode)
- `extractor/llm_extractor.py` - AI field extraction using Ollama (LLM mode)
- `config/tags.py` - Event type and topic definitions
- `utils/url_utils.py` - URL normalization and utilities
- `utils/io_utils.py` - JSONL I/O and logging

### Flow

#### LLM Mode Flow
1. **Discovery**: Use listing crawler to find event detail URLs
2. **Scraping**: Extract markdown content and images with Crawl4AI
3. **AI Extraction**: Use Ollama to extract and categorize structured fields
4. **Output**: Save enriched results to JSONL file

#### Scrapy Mode Flow  
1. **Integrated Extraction**: Scrapy spider handles discovery and extraction in one pass
2. **Rule-based Parsing**: CSS selectors and XPath extract structured data
3. **Output**: Save results to JSONL file (same format, faster processing)

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

### Mode Selection Issues

```bash
# Test Scrapy mode without AI dependencies
python main.py --url https://example.com/events --mode scrapy --test

# Test LLM mode (requires Ollama)
python main.py --url https://example.com/events --mode llm --test
```

### LLM Mode Issues (Ollama)

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Install/update Mistral model
ollama pull mistral

# Check available models
ollama list
```

### Scrapy Mode Issues

```bash
# Test Scrapy installation
python -c "import scrapy; print('Scrapy version:', scrapy.__version__)"

# Enable Scrapy debug logging
python main.py --url https://example.com/events --mode scrapy --verbose
```

### Common Problems

| Problem | LLM Mode | Scrapy Mode |
|---------|----------|-------------|
| **No events discovered** | Check link patterns in `link_pattern.py` | Same + check CSS selectors |
| **Scraping fails** | Some sites block Crawl4AI | Some sites block Scrapy |
| **Extraction fails** | Verify Ollama is running | Check CSS selector accuracy |
| **Import errors** | Install `crawl4ai`, `ollama` deps | Install `scrapy` dependency |
| **Slow performance** | Normal (AI processing) | Check concurrent request settings |

## Examples

### Mode Comparison

```bash
# Fast extraction for large datasets
python main.py --url https://events.example.com --mode scrapy --max-events 1000

# High-quality extraction with AI tagging
python main.py --url https://events.example.com --mode llm --max-events 50
```

### Test Mode (Discovery Only)

```bash
# Test Scrapy mode
python main.py --url https://events.example.com --mode scrapy --test

# Test LLM mode  
python main.py --url https://events.example.com --mode llm --test
```

### Production Examples

```bash
# Large-scale scraping with Scrapy
python main.py \
  --url https://events.example.com \
  --mode scrapy \
  --output events_2024.jsonl \
  --max-pages 50 \
  --max-events 5000 \
  --verbose

# Quality extraction with LLM
python main.py \
  --url https://events.example.com \
  --mode llm \
  --output curated_events.jsonl \
  --max-pages 10 \
  --max-events 200 \
  --ollama-url http://localhost:11434
```