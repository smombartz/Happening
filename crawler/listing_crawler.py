"""
Listing crawler for discovering event detail URLs from event listing pages.
"""
import time
import random
from typing import List, Set
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup

from link_pattern import is_event_link, is_next_link, should_exclude_link
from utils.url_utils import to_absolute, normalize, URLDeduplicator, get_domain, clean_url_for_display
from utils.io_utils import log_info, log_warn, log_error, log_debug


class ListingCrawler:
    """Crawler for discovering event detail URLs from listing pages."""
    
    def __init__(self, base_delay: float = 0.5, max_delay: float = 1.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.session = httpx.Client(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        )
        self.deduplicator = URLDeduplicator()
    
    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()
    
    def _sleep_politely(self):
        """Sleep for a random duration to be polite to servers."""
        delay = random.uniform(self.base_delay, self.max_delay)
        time.sleep(delay)
    
    def _fetch_page(self, url: str) -> str:
        """
        Fetch a single page and return HTML content.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string
            
        Raises:
            httpx.HTTPError: If request fails
        """
        log_debug(f"Fetching: {clean_url_for_display(url)}")
        
        try:
            response = self.session.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            log_error(f"Failed to fetch {clean_url_for_display(url)}: {e}")
            raise
    
    def _extract_links_from_html(self, html: str, base_url: str) -> List[dict]:
        """
        Extract all links from HTML content.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of link dictionaries with 'href', 'text', and 'rels'
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor.get('href', '').strip()
            if not href:
                continue
            
            # Get link text
            text = anchor.get_text(strip=True)
            
            # Get rel attributes
            rels = anchor.get('rel', [])
            if isinstance(rels, str):
                rels = [rels]
            
            # Convert to absolute URL
            absolute_url = to_absolute(base_url, href)
            if not absolute_url:
                continue
            
            links.append({
                'href': absolute_url,
                'text': text,
                'rels': rels
            })
        
        return links
    
    def _filter_event_links(self, links: List[dict], base_domain: str) -> List[str]:
        """
        Filter links to find likely event detail pages.
        
        Args:
            links: List of link dictionaries
            base_domain: Base domain to filter by
            
        Returns:
            List of event detail URLs
        """
        event_urls = []
        
        for link in links:
            href = link['href']
            text = link['text']
            
            # Skip if should exclude
            if should_exclude_link(href, text):
                continue
            
            # Only include links from same domain (optional filter)
            if base_domain and get_domain(href) != base_domain:
                log_debug(f"Skipping external link: {clean_url_for_display(href)}")
                continue
            
            # Check if it's an event link
            if is_event_link(href, text):
                normalized = normalize(href)
                if self.deduplicator.add_url(normalized):
                    event_urls.append(normalized)
                    log_debug(f"Found event link: {clean_url_for_display(normalized)} ({text[:50]}...)")
        
        return event_urls
    
    def _find_next_page_link(self, links: List[dict]) -> str:
        """
        Find the "next page" link for pagination.
        
        Args:
            links: List of link dictionaries
            
        Returns:
            Next page URL or empty string if not found
        """
        for link in links:
            href = link['href']
            text = link['text']
            rels = link['rels']
            
            if is_next_link(href, text, rels):
                log_debug(f"Found next page link: {clean_url_for_display(href)} ({text})")
                return normalize(href)
        
        return ""
    
    def discover_detail_urls(self, listing_url: str, max_pages: int = 10) -> List[str]:
        """
        Discover event detail URLs from a listing page with pagination support.
        
        Args:
            listing_url: The listing page URL to start from
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of unique event detail URLs
        """
        all_event_urls = []
        current_url = normalize(listing_url)
        base_domain = get_domain(current_url)
        pages_crawled = 0
        visited_urls = set()
        
        log_info(f"Starting discovery from: {clean_url_for_display(current_url)}")
        
        while current_url and pages_crawled < max_pages:
            # Avoid infinite loops
            if current_url in visited_urls:
                log_warn(f"Already visited {clean_url_for_display(current_url)}, stopping")
                break
            
            visited_urls.add(current_url)
            pages_crawled += 1
            
            try:
                # Fetch the page
                html = self._fetch_page(current_url)
                log_info(f"Crawled page {pages_crawled}: {clean_url_for_display(current_url)}")
                
                # Extract all links
                links = self._extract_links_from_html(html, current_url)
                log_debug(f"Found {len(links)} total links")
                
                # Filter for event links
                event_urls = self._filter_event_links(links, base_domain)
                all_event_urls.extend(event_urls)
                log_info(f"Found {len(event_urls)} event links on page {pages_crawled}")
                
                # Find next page link (only if we haven't reached max pages)
                if pages_crawled < max_pages:
                    next_url = self._find_next_page_link(links)
                    if next_url:
                        current_url = next_url
                        log_info(f"Found next page: {clean_url_for_display(next_url)}")
                        # Sleep before next request
                        self._sleep_politely()
                    else:
                        log_info("No next page found, pagination complete")
                        break
                else:
                    break
                
            except Exception as e:
                log_error(f"Error crawling page {pages_crawled} ({clean_url_for_display(current_url)}): {e}")
                break
        
        unique_count = len(set(all_event_urls))
        log_info(f"Discovery complete: {unique_count} unique event URLs found across {pages_crawled} pages")
        
        # Return deduplicated list
        return list(dict.fromkeys(all_event_urls))


def discover_detail_urls(listing_url: str, max_pages: int = 10) -> List[str]:
    """
    Convenience function to discover event detail URLs.
    
    Args:
        listing_url: The listing page URL to start from
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List of unique event detail URLs
    """
    crawler = ListingCrawler()
    return crawler.discover_detail_urls(listing_url, max_pages)