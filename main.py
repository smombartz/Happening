#!/usr/bin/env python3
"""
Main CLI entrypoint for the event scraper.

Usage:
    python main.py --url <listing_url> [--output <path.jsonl>] [--test] [--max-pages <int>] [--verbose]
"""
import argparse
import sys
import asyncio
from pathlib import Path

from crawler.listing_crawler import discover_detail_urls
from crawler.detail_scraper import scrape_markdown_sync
from extractor.llm_extractor import extract
from utils.io_utils import (
    setup_logging, log_info, log_warn, log_error, 
    write_jsonl, count_jsonl_lines, ensure_file_empty, print_progress
)
from utils.url_utils import clean_url_for_display


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape events from listing pages and extract structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover and scrape events
  python main.py --url https://example.com/events
  
  # Test mode (discovery only)
  python main.py --url https://example.com/events --test
  
  # Custom output file and pagination limit
  python main.py --url https://example.com/events --output my_events.jsonl --max-pages 5
  
  # Verbose logging
  python main.py --url https://example.com/events --verbose
        """
    )
    
    parser.add_argument(
        '--url',
        required=True,
        help='Listing URL to scrape events from'
    )
    
    parser.add_argument(
        '--output',
        default='events.jsonl',
        help='Output JSONL file path (default: events.jsonl)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: discover and print detail URLs only, no scraping'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=10,
        help='Maximum pages to crawl for discovery (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--model',
        default='mistral',
        help='Ollama model name (default: mistral)'
    )
    
    return parser.parse_args()


def check_ollama_availability(ollama_url: str, model_name: str) -> bool:
    """
    Check if Ollama is running and the model is available.
    
    Args:
        ollama_url: Ollama API URL
        model_name: Model name to check
        
    Returns:
        True if available, False otherwise
    """
    try:
        import httpx
        
        # Check if Ollama is running
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            
            # Check if model is available
            models = response.json()
            available_models = [model['name'].split(':')[0] for model in models.get('models', [])]
            
            if model_name not in available_models:
                log_error(f"Model '{model_name}' not found. Available models: {', '.join(available_models)}")
                log_info(f"To install the model, run: ollama pull {model_name}")
                return False
            
            log_info(f"Ollama is running with model '{model_name}'")
            return True
            
    except Exception as e:
        log_error(f"Ollama not available at {ollama_url}: {e}")
        log_info("Make sure Ollama is running: https://ollama.ai/")
        return False


def main():
    """Main CLI function."""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    log_info(f"Starting event scraper for: {clean_url_for_display(args.url)}")
    
    try:
        # Step 1: Discovery
        log_info(f"Discovering event URLs (max {args.max_pages} pages)...")
        detail_urls = discover_detail_urls(args.url, max_pages=args.max_pages)
        
        if not detail_urls:
            log_warn("No event URLs discovered")
            return 1
        
        log_info(f"Discovered {len(detail_urls)} unique event URLs")
        
        # Test mode: just print URLs and exit
        if args.test:
            log_info("Test mode: printing discovered URLs")
            for i, url in enumerate(detail_urls, 1):
                print(f"{i:3d}. {url}")
            return 0
        
        # Check Ollama availability before scraping
        if not check_ollama_availability(args.ollama_url, args.model):
            return 1
        
        # Prepare output file
        output_path = Path(args.output)
        if output_path.exists():
            existing_count = count_jsonl_lines(str(output_path))
            log_info(f"Output file exists with {existing_count} existing records")
            
            # Ask user if they want to append or overwrite
            try:
                choice = input("Append to existing file? (y/n): ").lower().strip()
                if choice not in ['y', 'yes']:
                    ensure_file_empty(str(output_path))
                    log_info("Output file cleared")
            except KeyboardInterrupt:
                log_info("Cancelled by user")
                return 1
        else:
            ensure_file_empty(str(output_path))
        
        # Step 2: Scraping and Extraction
        log_info(f"Starting scraping and extraction for {len(detail_urls)} URLs...")
        
        from extractor.llm_extractor import LLMExtractor
        extractor = LLMExtractor(ollama_url=args.ollama_url, model_name=args.model)
        
        successful_extractions = 0
        failed_extractions = 0
        
        for i, detail_url in enumerate(detail_urls, 1):
            try:
                print_progress(i, len(detail_urls), "Processing: ")
                
                # Scrape content
                log_info(f"Scraping {i}/{len(detail_urls)}: {clean_url_for_display(detail_url)}")
                scrape_result = scrape_markdown_sync(detail_url)
                
                if not scrape_result['content_markdown']:
                    log_warn(f"No content scraped from {clean_url_for_display(detail_url)}")
                    failed_extractions += 1
                    continue
                
                # Extract structured data
                log_info(f"Extracting data from {clean_url_for_display(detail_url)}")
                extraction = extractor.extract(
                    markdown=scrape_result['content_markdown'],
                    source_url=args.url,
                    detail_url=detail_url,
                    image_hint=scrape_result['image_url']
                )
                
                # Create final record
                record = {
                    'source_url': args.url,
                    'detail_url': detail_url,
                    'content_markdown': scrape_result['content_markdown'],
                    'extraction': extraction
                }
                
                # Write to output file
                write_jsonl(str(output_path), record)
                successful_extractions += 1
                
                log_info(f"Saved: {extraction.get('title', 'No title')} [{i}/{len(detail_urls)}]")
                
            except KeyboardInterrupt:
                log_info("Cancelled by user")
                break
                
            except Exception as e:
                log_error(f"Failed to process {clean_url_for_display(detail_url)}: {e}")
                failed_extractions += 1
                continue
        
        # Summary
        log_info("=" * 60)
        log_info("SCRAPING COMPLETE")
        log_info(f"Total URLs discovered: {len(detail_urls)}")
        log_info(f"Successful extractions: {successful_extractions}")
        log_info(f"Failed extractions: {failed_extractions}")
        log_info(f"Output file: {output_path.absolute()}")
        
        final_count = count_jsonl_lines(str(output_path))
        log_info(f"Total records in output file: {final_count}")
        
        return 0
        
    except KeyboardInterrupt:
        log_info("Cancelled by user")
        return 1
        
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())