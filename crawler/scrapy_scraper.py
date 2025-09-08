"""
Scrapy-based event scraper for rule-based extraction without LLM dependency.
"""
import json
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from datetime import datetime
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import tempfile
import os

from utils.io_utils import log_info, log_warn, log_error, log_debug
from utils.url_utils import normalize, get_domain, clean_url_for_display
from link_pattern import is_event_link, should_exclude_link


class EventItem(scrapy.Item):
    """Scrapy item for event data."""
    title = scrapy.Field()
    start_date = scrapy.Field()
    end_date = scrapy.Field()
    start_time = scrapy.Field()
    end_time = scrapy.Field()
    timezone = scrapy.Field()
    location_address = scrapy.Field()
    image_url = scrapy.Field()
    description = scrapy.Field()
    event_type = scrapy.Field()
    topics = scrapy.Field()
    source_url = scrapy.Field()
    detail_url = scrapy.Field()


class EventSpider(scrapy.Spider):
    """Scrapy spider for extracting event data using CSS/XPath selectors."""
    
    name = 'events'
    
    def __init__(self, start_url=None, max_pages=10, max_events=None, *args, **kwargs):
        super(EventSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url] if start_url else []
        self.max_pages = int(max_pages)
        self.max_events = int(max_events) if max_events else None
        self.pages_crawled = 0
        self.events_found = 0
        self.domain = get_domain(start_url) if start_url else ""
        self.results = []
        
    def parse(self, response):
        """Parse listing page to find event detail links."""
        self.pages_crawled += 1
        log_debug(f"Parsing listing page {self.pages_crawled}: {clean_url_for_display(response.url)}")
        
        # Extract event links using existing link pattern logic
        event_links = []
        for link in response.css('a[href]'):
            href = link.attrib.get('href', '').strip()
            text = link.css('::text').get('').strip()
            
            if not href:
                continue
                
            absolute_url = response.urljoin(href)
            
            # Apply existing filters
            if should_exclude_link(absolute_url, text):
                continue
                
            # Only same domain
            if self.domain and get_domain(absolute_url) != self.domain:
                continue
                
            # Check if it's an event link
            if is_event_link(absolute_url, text):
                event_links.append(absolute_url)
        
        # Remove duplicates and limit
        event_links = list(dict.fromkeys(event_links))
        
        # Apply max_events limit
        if self.max_events and self.events_found + len(event_links) > self.max_events:
            remaining = self.max_events - self.events_found
            event_links = event_links[:remaining]
        
        log_debug(f"Found {len(event_links)} event links on page {self.pages_crawled}")
        
        # Follow event detail pages
        for url in event_links:
            if self.max_events and self.events_found >= self.max_events:
                break
            self.events_found += 1
            yield response.follow(url, self.parse_event, meta={'source_url': response.url})
        
        # Follow pagination if not at max pages
        if self.pages_crawled < self.max_pages:
            # Look for next page links
            next_selectors = [
                'a[rel="next"]',
                'a:contains("Next")',  
                'a:contains(">")',
                '.pagination a:last-child',
                '.pager-next a',
            ]
            
            for selector in next_selectors:
                next_link = response.css(selector + '::attr(href)').get()
                if next_link:
                    next_url = response.urljoin(next_link)
                    log_debug(f"Following next page: {clean_url_for_display(next_url)}")
                    yield response.follow(next_url, self.parse)
                    break
    
    def parse_event(self, response):
        """Parse individual event page to extract structured data."""
        log_debug(f"Parsing event: {clean_url_for_display(response.url)}")
        
        item = EventItem()
        
        # Extract title
        title_selectors = [
            'h1::text',
            '.event-title::text',
            '.title::text',
            'title::text',
            '[property="og:title"]::attr(content)',
        ]
        item['title'] = self._extract_first(response, title_selectors)
        
        # Extract description  
        desc_selectors = [
            '.event-description::text',
            '.description p::text',
            '.content p::text',
            '[property="og:description"]::attr(content)',
            'meta[name="description"]::attr(content)',
        ]
        description = self._extract_first(response, desc_selectors)
        if description and len(description) > 500:
            description = description[:497] + "..."
        item['description'] = description
        
        # Extract location
        location_selectors = [
            '.event-location::text',
            '.location::text',
            '.venue::text',
            '.address::text',
        ]
        item['location_address'] = self._extract_first(response, location_selectors)
        
        # Extract image
        image_selectors = [
            '[property="og:image"]::attr(content)',
            '[name="twitter:image"]::attr(content)',
            '.event-image img::attr(src)',
            'img::attr(src)',
        ]
        image_url = self._extract_first(response, image_selectors)
        if image_url:
            item['image_url'] = response.urljoin(image_url)
        
        # Extract dates and times
        self._extract_datetime_info(response, item)
        
        # Set fields that require LLM (not available in Scrapy mode)
        item['event_type'] = None
        item['topics'] = []
        item['timezone'] = None
        
        # Set URLs
        item['source_url'] = response.meta.get('source_url', '')
        item['detail_url'] = response.url
        
        # Store result for later retrieval
        result_dict = dict(item)
        # Convert None values to proper nulls for JSON
        for key, value in result_dict.items():
            if value is None or value == "":
                result_dict[key] = None
        
        self.results.append(result_dict)
        
        log_debug(f"Extracted event: {result_dict.get('title', 'Untitled')}")
        yield item
    
    def _extract_first(self, response, selectors):
        """Extract first non-empty result from list of CSS selectors."""
        for selector in selectors:
            try:
                result = response.css(selector).get()
                if result and result.strip():
                    return result.strip()
            except Exception:
                continue
        return None
    
    def _extract_datetime_info(self, response, item):
        """Extract date and time information using various strategies."""
        
        # Look for structured data first
        json_ld_scripts = response.css('script[type="application/ld+json"]::text').getall()
        for script in json_ld_scripts:
            try:
                data = json.loads(script)
                if isinstance(data, dict) and data.get('@type') == 'Event':
                    start_date = data.get('startDate')
                    end_date = data.get('endDate')
                    
                    if start_date:
                        item['start_date'], item['start_time'] = self._parse_iso_datetime(start_date)
                    if end_date:
                        item['end_date'], item['end_time'] = self._parse_iso_datetime(end_date)
                    return
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Fall back to text extraction
        datetime_selectors = [
            '.event-date::text',
            '.date::text',
            '.event-time::text',
            '.time::text',
            'time::attr(datetime)',
            'time::text',
        ]
        
        datetime_text = ""
        for selector in datetime_selectors:
            result = response.css(selector).get()
            if result:
                datetime_text += " " + result.strip()
        
        if datetime_text.strip():
            self._parse_datetime_text(datetime_text.strip(), item)
    
    def _parse_iso_datetime(self, iso_string):
        """Parse ISO datetime string into date and time components."""
        try:
            dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M')
            return date_str, time_str
        except ValueError:
            return None, None
    
    def _parse_datetime_text(self, text, item):
        """Parse date/time from free text using regex patterns."""
        
        # Simple date patterns
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # 2025-08-08
            r'(\d{1,2}/\d{1,2}/\d{4})',  # 8/8/2025
            r'(\w+ \d{1,2}, \d{4})',  # August 8, 2025
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                try:
                    if '-' in date_str:
                        # Already in correct format
                        item['start_date'] = date_str
                    elif '/' in date_str:
                        # Convert from MM/DD/YYYY to YYYY-MM-DD
                        dt = datetime.strptime(date_str, '%m/%d/%Y')
                        item['start_date'] = dt.strftime('%Y-%m-%d')
                    else:
                        # Parse natural language date
                        dt = datetime.strptime(date_str, '%B %d, %Y')
                        item['start_date'] = dt.strftime('%Y-%m-%d')
                    
                    # If only single date found, set end_date = start_date
                    if not item.get('end_date'):
                        item['end_date'] = item['start_date']
                    break
                except ValueError:
                    continue
        
        # Simple time patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',  # 7:00 PM
            r'(\d{1,2}:\d{2})',  # 19:00
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                time_str = match.group(1)
                try:
                    if 'AM' in time_str.upper() or 'PM' in time_str.upper():
                        # Convert to 24-hour format
                        dt = datetime.strptime(time_str.upper(), '%I:%M %p')
                        item['start_time'] = dt.strftime('%H:%M')
                    else:
                        # Assume already 24-hour format
                        item['start_time'] = time_str
                    break
                except ValueError:
                    continue


class ScrapyEventScraper:
    """Wrapper class for Scrapy-based event scraping."""
    
    def __init__(self):
        pass
    
    def scrape_events(self, listing_url: str, max_pages: int = 10, max_events: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scrape events using Scrapy spider.
        
        Args:
            listing_url: URL of the listing page
            max_pages: Maximum number of listing pages to crawl
            max_events: Maximum number of events to extract
            
        Returns:
            List of extracted event dictionaries
        """
        log_info(f"Starting Scrapy extraction from {clean_url_for_display(listing_url)}")
        
        # Configure Scrapy settings
        settings = get_project_settings()
        settings.update({
            'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'ROBOTSTXT_OBEY': False,
            'DOWNLOAD_DELAY': 1,
            'RANDOMIZE_DOWNLOAD_DELAY': True,
            'CONCURRENT_REQUESTS': 1,
            'COOKIES_ENABLED': True,
            'LOG_LEVEL': 'WARNING',  # Reduce Scrapy logging noise
        })
        
        # Create a temporary results holder
        results = []
        
        # Custom spider class that stores results
        class ResultsSpider(EventSpider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.results_holder = results
            
            def parse_event(self, response):
                item_generator = super().parse_event(response)
                for item in item_generator:
                    # Convert scrapy item to dict and store
                    if isinstance(item, EventItem):
                        result_dict = dict(item)
                        # Ensure consistent None handling
                        for key, value in result_dict.items():
                            if value is None or value == "":
                                result_dict[key] = None
                        self.results_holder.append(result_dict)
                    yield item
        
        # Create and run crawler process
        process = CrawlerProcess(settings)
        
        process.crawl(
            ResultsSpider,
            start_url=listing_url,
            max_pages=max_pages,
            max_events=max_events
        )
        
        process.start()
        
        log_info(f"Scrapy extraction complete: {len(results)} events found")
        return results


def scrape_with_scrapy(listing_url: str, max_pages: int = 10, max_events: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to scrape events with Scrapy.
    
    Args:
        listing_url: URL of the listing page
        max_pages: Maximum number of listing pages to crawl  
        max_events: Maximum number of events to extract
        
    Returns:
        List of extracted event dictionaries
    """
    scraper = ScrapyEventScraper()
    return scraper.scrape_events(listing_url, max_pages, max_events)