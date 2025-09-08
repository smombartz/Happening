"""
URL utilities for normalization, absolute URL resolution, and deduplication.
"""
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from typing import Set


def to_absolute(base_url: str, href: str) -> str:
    """
    Convert a relative URL to absolute using base URL.
    
    Args:
        base_url: The base URL to resolve against
        href: The href (can be relative or absolute)
        
    Returns:
        Absolute URL
    """
    if not href:
        return ""
    
    # Handle fragment-only links and empty hrefs
    if href.startswith("#") or href == "/":
        return ""
    
    return urljoin(base_url, href)


def normalize(url: str) -> str:
    """
    Normalize URL by:
    - Lowercasing scheme and hostname
    - Removing fragments (#section)
    - Removing UTM parameters and common tracking params
    - Normalizing trailing slashes
    
    Args:
        url: The URL to normalize
        
    Returns:
        Normalized URL
    """
    if not url:
        return ""
    
    parsed = urlparse(url)
    
    # Lowercase scheme and hostname
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    
    # Remove fragments
    fragment = ""
    
    # Parse and filter query parameters
    query_params = parse_qs(parsed.query, keep_blank_values=False)
    
    # Remove tracking parameters
    tracking_params = {
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
        'gclid', 'fbclid', 'igshid', 'mc_cid', 'mc_eid',
        '_ga', '_gl', 'ref', 'referrer', 'source'
    }
    
    filtered_params = {
        key: value for key, value in query_params.items() 
        if key.lower() not in tracking_params
    }
    
    # Rebuild query string
    if filtered_params:
        # Flatten list values and sort for consistency
        flat_params = []
        for key, values in sorted(filtered_params.items()):
            for value in values:
                flat_params.append((key, value))
        query = urlencode(flat_params)
    else:
        query = ""
    
    # Normalize path - remove trailing slash except for root
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    
    # Reconstruct URL
    normalized = urlunparse((
        scheme,
        netloc,
        path,
        parsed.params,
        query,
        fragment
    ))
    
    return normalized


def get_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        Domain name (lowercase)
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs are from the same domain.
    
    Args:
        url1: First URL
        url2: Second URL
        
    Returns:
        True if same domain
    """
    return get_domain(url1) == get_domain(url2)


class URLDeduplicator:
    """Helper class for deduplicating URLs during crawling."""
    
    def __init__(self):
        self.seen_urls: Set[str] = set()
    
    def add_url(self, url: str) -> bool:
        """
        Add URL to seen set.
        
        Args:
            url: URL to add
            
        Returns:
            True if URL was newly added, False if already seen
        """
        normalized = normalize(url)
        if normalized in self.seen_urls:
            return False
        
        self.seen_urls.add(normalized)
        return True
    
    def has_seen(self, url: str) -> bool:
        """
        Check if URL has been seen before.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL was already seen
        """
        normalized = normalize(url)
        return normalized in self.seen_urls
    
    def get_seen_count(self) -> int:
        """Get count of unique URLs seen."""
        return len(self.seen_urls)


def clean_url_for_display(url: str, max_length: int = 80) -> str:
    """
    Clean URL for display purposes (logging, etc.).
    
    Args:
        url: URL to clean
        max_length: Maximum length for display
        
    Returns:
        Cleaned URL suitable for display
    """
    if not url:
        return ""
    
    # Remove protocol for cleaner display
    display_url = url
    if display_url.startswith("http://"):
        display_url = display_url[7:]
    elif display_url.startswith("https://"):
        display_url = display_url[8:]
    
    # Truncate if too long
    if len(display_url) > max_length:
        display_url = display_url[:max_length-3] + "..."
    
    return display_url