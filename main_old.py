#!/usr/bin/env python3
"""
Main CLI entrypoint for the event scraper.

Usage:
    python main.py --url <listing_url> [--output <path.jsonl>] [--test] [--max-pages <int>] [--verbose]
"""
import argparse
import sys
import asyncio
import time
from pathlib import Path

from crawler.listing_crawler import discover_detail_urls
from crawler.detail_scraper import scrape_markdown_sync
from extractor.llm_extractor import extract
from utils.io_utils import (
    setup_logging, log_info, log_warn, log_error, 
    write_jsonl, count_jsonl_lines, ensure_file_empty, print_progress,
    print_progress_bar, print_status, print_substep, print_final_substep,
    print_section_header, print_timer, print_stats, print_spinner_message,
    print_spinner_complete
)
from utils.url_utils import clean_url_for_display


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape events from listing pages and extract structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LLM mode (default) - AI extraction with event types and topics
  python main.py --url https://example.com/events --mode llm
  
  # Scrapy mode - Fast rule-based extraction (no AI)  
  python main.py --url https://example.com/events --mode scrapy
  
  # Limit to 2 listing pages, process all discovered events
  python main.py --url https://example.com/events --max-pages 2
  
  # Limit to 2 listing pages, process only first 5 events
  python main.py --url https://example.com/events --max-pages 2 --max-events 5
  
  # Test mode (discovery only)
  python main.py --url https://example.com/events --test
  
  # Custom output file with verbose logging
  python main.py --url https://example.com/events --output my_events.jsonl --verbose
        """
    )
    
    parser.add_argument(
        '--url',
        required=True,
        help='Listing URL to scrape events from'
    )
    
    parser.add_argument(
        '--mode',
        choices=['llm', 'scrapy'],
        default='llm',
        help='Scraping mode: llm (AI extraction) or scrapy (rule-based extraction, default: llm)'
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
        help='Maximum listing pages to crawl for discovery (default: 10)'
    )
    
    parser.add_argument(
        '--max-events',
        type=int,
        default=None,
        help='Maximum number of individual events to process (default: unlimited)'
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
    start_time = time.time()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    print_section_header("Event Scraper Starting")
    print_substep(f"Target URL: {clean_url_for_display(args.url)}")
    print_substep(f"Scraping mode: {args.mode.upper()}")
    print_substep(f"Max listing pages: {args.max_pages}")
    if args.max_events:
        print_substep(f"Max events to process: {args.max_events}")
    else:
        print_substep(f"Max events to process: unlimited")
    print_final_substep(f"Output file: {args.output}")
    
    try:
        # Branch based on scraping mode
        if args.mode == 'scrapy':
            return run_scrapy_mode(args, start_time)
        else:
            return run_llm_mode(args, start_time)
        
        if not detail_urls:
            print_final_substep("No event URLs discovered", status="✗")
            return 1
        
        print_final_substep(f"Discovery complete: {len(detail_urls)} unique events found")
        
        # Apply max-events limit to discovered URLs
        if args.max_events and len(detail_urls) > args.max_events:
            detail_urls = detail_urls[:args.max_events]
            print_substep(f"Limited to first {args.max_events} events (--max-events)", status="⚠️")
        
        print_timer("Discovery", discovery_time)
        
        # Test mode: just print URLs and exit
        if args.test:
            print_status("Test Mode - Discovered URLs", "📋")
            for i, url in enumerate(detail_urls, 1):
                print(f"{i:3d}. {url}")
            
            stats = {
                "total_urls_discovered": len(detail_urls),
                "discovery_time": f"{discovery_time:.1f}s"
            }
            if args.max_events:
                stats["max_events_limit"] = args.max_events
            if args.max_pages != 10:  # Only show if not default
                stats["max_pages_limit"] = args.max_pages
            
            print_stats(stats, "Discovery Summary")
            return 0
        
        # Check Ollama availability before scraping
        print_substep("Checking Ollama availability")
        if not check_ollama_availability(args.ollama_url, args.model):
            print_final_substep("Ollama check failed", status="✗")
            return 1
        print_substep("Ollama ready", status="✓")
        
        # Prepare output file
        print_substep("Preparing output file")
        output_path = Path(args.output)
        if output_path.exists():
            existing_count = count_jsonl_lines(str(output_path))
            print_substep(f"Output file exists with {existing_count} existing records")
            
            # Ask user if they want to append or overwrite
            try:
                choice = input("Append to existing file? (y/n): ").lower().strip()
                if choice not in ['y', 'yes']:
                    ensure_file_empty(str(output_path))
                    print_substep("Output file cleared", status="✓")
                else:
                    print_substep("Will append to existing file", status="✓")
            except KeyboardInterrupt:
                print_final_substep("Cancelled by user", status="✗")
                return 1
        else:
            ensure_file_empty(str(output_path))
            print_substep("Created new output file", status="✓")
        
        # Step 2: Scraping and Extraction
        print_status("Scraping and Extracting", "🔄")
        print_substep(f"Will process {len(detail_urls)} events")
        
        processing_start = time.time()
        
        from extractor.llm_extractor import LLMExtractor
        extractor = LLMExtractor(ollama_url=args.ollama_url, model_name=args.model)
        
        successful_extractions = 0
        failed_extractions = 0
        processing_times = []
        
        for i, detail_url in enumerate(detail_urls, 1):
            try:
                item_start = time.time()
                
                # Calculate ETA and processing rate
                if i > 1:
                    avg_time_per_item = sum(processing_times) / len(processing_times)
                    remaining_items = len(detail_urls) - i + 1
                    eta_seconds = int(avg_time_per_item * remaining_items)
                    rate = 60 / avg_time_per_item if avg_time_per_item > 0 else 0
                else:
                    eta_seconds = None
                    rate = None
                
                # Show enhanced progress bar
                print_progress_bar(i, len(detail_urls), 
                                 prefix=f"Processing ({i}/{len(detail_urls)}):",
                                 rate=rate, eta_seconds=eta_seconds)
                
                # Scrape content
                print_substep(f"Scraping: {clean_url_for_display(detail_url, 50)}", status="⏳")
                scrape_result = scrape_markdown_sync(detail_url)
                
                if not scrape_result['content_markdown']:
                    print_substep(f"No content scraped", status="✗")
                    failed_extractions += 1
                    continue
                
                content_length = len(scrape_result['content_markdown'])
                image_status = "Image found" if scrape_result['image_url'] else "No image"
                print_substep(f"Content extracted: {content_length:,} chars, {image_status}", status="✓")
                
                # Extract structured data
                print_substep("Calling LLM for field extraction", status="⏳")
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
                
                event_title = extraction.get('title', 'Untitled Event')
                event_date = extraction.get('start_date', 'No date')
                print_final_substep(f"Saved: '{event_title}' ({event_date})", status="✓")
                
                # Track processing time
                item_time = time.time() - item_start
                processing_times.append(item_time)
                
                # Keep only last 10 times for better ETA accuracy
                if len(processing_times) > 10:
                    processing_times = processing_times[-10:]
                
            except KeyboardInterrupt:
                print_final_substep("Cancelled by user", status="✗")
                break
                
            except Exception as e:
                print_substep(f"Failed: {str(e)[:60]}...", status="✗")
                failed_extractions += 1
                continue
        
        # Summary
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        print_status("Processing Complete", "✅")
        print_timer("Processing", processing_time)
        print_timer("Total Runtime", total_time)
        
        final_count = count_jsonl_lines(str(output_path))
        success_rate = (successful_extractions / len(detail_urls)) * 100 if detail_urls else 0
        
        final_stats = {
            "events_processed": len(detail_urls),
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
            "success_rate": f"{success_rate:.1f}%",
            "final_records_in_file": final_count,
            "avg_processing_time": f"{processing_time/len(detail_urls):.1f}s per item" if detail_urls else "N/A"
        }
        
        # Add limit information if applied
        if args.max_events:
            final_stats["max_events_limit"] = f"{args.max_events} (applied)"
        if args.max_pages != 10:
            final_stats["max_pages_limit"] = args.max_pages
        
        print_stats(final_stats, "Final Summary")
        
        print_final_substep(f"Results saved to: {output_path.absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        print_status("Cancelled by user", "⚠️")
        if 'successful_extractions' in locals() and successful_extractions > 0:
            print_substep(f"Partial results: {successful_extractions} events saved before cancellation")
        return 1
        
    except Exception as e:
        print_status("Unexpected Error", "❌")
        print_substep(f"Error: {str(e)}", status="✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())