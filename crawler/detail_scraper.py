"""
Detail scraper for extracting content from individual event pages using Crawl4AI.
"""
import time
import random
from typing import Dict, Optional
import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

from utils.io_utils import log_info, log_warn, log_error, log_debug
from utils.url_utils import clean_url_for_display


class DetailScraper:
    """Scraper for extracting content from individual event detail pages."""
    
    def __init__(self, base_delay: float = 0.5, max_delay: float = 1.0, max_retries: int = 3):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.crawler = None
    
    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(verbose=False)
        await self.crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    def _sleep_politely(self):
        """Sleep for a random duration to be polite to servers."""
        delay = random.uniform(self.base_delay, self.max_delay)
        time.sleep(delay)
    
    def _extract_best_image(self, html: str, url: str) -> Optional[str]:
        """
        Extract the best image URL from HTML content.
        
        Priority order:
        1. og:image meta tag
        2. twitter:image meta tag  
        3. First image in main/article content
        4. First img tag on page
        
        Args:
            html: HTML content
            url: Page URL for resolving relative URLs
            
        Returns:
            Best image URL or None
        """
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 1. Check og:image
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                image_url = og_image['content'].strip()
                if image_url:
                    log_debug(f"Found og:image: {image_url}")
                    return self._resolve_image_url(image_url, url)
            
            # 2. Check twitter:image
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                image_url = twitter_image['content'].strip()
                if image_url:
                    log_debug(f"Found twitter:image: {image_url}")
                    return self._resolve_image_url(image_url, url)
            
            # 3. Check first image in main content areas
            main_selectors = ['main', 'article', '.content', '.main-content', '#main', '#content']
            for selector in main_selectors:
                main_area = soup.select_one(selector)
                if main_area:
                    img = main_area.find('img', src=True)
                    if img and img.get('src'):
                        image_url = img['src'].strip()
                        if image_url and not self._is_icon_or_logo(img):
                            log_debug(f"Found main content image: {image_url}")
                            return self._resolve_image_url(image_url, url)
            
            # 4. Fallback to first non-icon image on page
            for img in soup.find_all('img', src=True):
                image_url = img.get('src', '').strip()
                if image_url and not self._is_icon_or_logo(img):
                    log_debug(f"Found fallback image: {image_url}")
                    return self._resolve_image_url(image_url, url)
            
            log_debug("No suitable images found")
            return None
            
        except Exception as e:
            log_error(f"Error extracting image from {clean_url_for_display(url)}: {e}")
            return None
    
    def _is_icon_or_logo(self, img_tag) -> bool:
        """
        Check if an image tag is likely an icon or logo (should be ignored).
        
        Args:
            img_tag: BeautifulSoup img tag
            
        Returns:
            True if likely an icon/logo
        """
        src = img_tag.get('src', '').lower()
        alt = img_tag.get('alt', '').lower()
        class_attr = ' '.join(img_tag.get('class', [])).lower()
        
        # Check dimensions if available
        width = img_tag.get('width')
        height = img_tag.get('height')
        
        if width and height:
            try:
                w, h = int(width), int(height)
                # Skip very small images (likely icons)
                if w <= 50 or h <= 50:
                    return True
            except (ValueError, TypeError):
                pass
        
        # Check for icon/logo indicators
        icon_indicators = [
            'icon', 'logo', 'favicon', 'avatar', 'profile', 'badge',
            'button', 'arrow', 'star', 'rating', 'social'
        ]
        
        for indicator in icon_indicators:
            if (indicator in src or 
                indicator in alt or 
                indicator in class_attr):
                return True
        
        return False
    
    def _resolve_image_url(self, image_url: str, base_url: str) -> str:
        """
        Resolve image URL to absolute URL.
        
        Args:
            image_url: Image URL (may be relative)
            base_url: Base URL to resolve against
            
        Returns:
            Absolute image URL
        """
        from urllib.parse import urljoin, urlparse
        
        if not image_url:
            return ""
        
        # Already absolute
        if image_url.startswith(('http://', 'https://')):
            return image_url
        
        # Protocol-relative
        if image_url.startswith('//'):
            parsed_base = urlparse(base_url)
            return f"{parsed_base.scheme}:{image_url}"
        
        # Relative - resolve against base
        return urljoin(base_url, image_url)
    
    async def scrape_markdown(self, url: str) -> Dict[str, Optional[str]]:
        """
        Scrape a single event detail page and extract markdown content.
        
        Args:
            url: URL of the event detail page
            
        Returns:
            Dictionary with 'content_markdown' and 'image_url' keys
        """
        log_debug(f"Scraping: {clean_url_for_display(url)}")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    log_warn(f"Retry {attempt + 1}/{self.max_retries} for {clean_url_for_display(url)}")
                    time.sleep(1.0)  # Fixed delay between retries
                
                # Use Crawl4AI to scrape
                result = await self.crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    bypass_cache=True,
                    exclude_external_links=True,
                    exclude_social_media_links=True,
                    remove_overlay_elements=True
                )
                
                if result.success and result.markdown:
                    # Extract best image from HTML
                    image_url = self._extract_best_image(result.html, url)
                    
                    log_info(f"Scraped {clean_url_for_display(url)} - {len(result.markdown)} chars")
                    
                    return {
                        'content_markdown': result.markdown.strip(),
                        'image_url': image_url
                    }
                else:
                    error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown error"
                    log_error(f"Crawl4AI failed for {clean_url_for_display(url)}: {error_msg}")
                    
                    if attempt == self.max_retries - 1:
                        return {
                            'content_markdown': "",
                            'image_url': None
                        }
                    
            except Exception as e:
                log_error(f"Exception scraping {clean_url_for_display(url)} (attempt {attempt + 1}): {e}")
                
                if attempt == self.max_retries - 1:
                    return {
                        'content_markdown': "",
                        'image_url': None
                    }
        
        return {
            'content_markdown': "",
            'image_url': None
        }


async def scrape_markdown(url: str) -> Dict[str, Optional[str]]:
    """
    Convenience function to scrape a single URL.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary with 'content_markdown' and 'image_url' keys
    """
    async with DetailScraper() as scraper:
        return await scraper.scrape_markdown(url)


def scrape_markdown_sync(url: str) -> Dict[str, Optional[str]]:
    """
    Synchronous wrapper for scraping a single URL.
    
    Args:
        url: URL to scrape
        
    Returns:
        Dictionary with 'content_markdown' and 'image_url' keys
    """
    return asyncio.run(scrape_markdown(url))


async def scrape_multiple_urls(urls: list, max_concurrent: int = 3) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Scrape multiple URLs concurrently with rate limiting.
    
    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        Dictionary mapping URLs to their scrape results
    """
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_semaphore(url: str) -> tuple:
        async with semaphore:
            async with DetailScraper() as scraper:
                result = await scraper.scrape_markdown(url)
                return url, result
    
    # Run all scraping tasks
    tasks = [scrape_with_semaphore(url) for url in urls]
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in completed_results:
        if isinstance(result, Exception):
            log_error(f"Task failed with exception: {result}")
            continue
        
        url, scrape_result = result
        results[url] = scrape_result
    
    return results