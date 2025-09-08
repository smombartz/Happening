"""
Centralized link extraction patterns and heuristics for event discovery.
"""
import re
from typing import List

EVENT_LINK_SUBSTRINGS = [
    "event", "events", "calendar", "whatson", "whats-on", 
    "program", "programs", "happening", "show", "shows",
    "performance", "performances", "concert", "concerts"
]

DATE_SLUG_REGEXES = [
    re.compile(r"/20\d{2}/\d{1,2}/\d{1,2}/"),
    re.compile(r"/\d{4}-\d{2}-\d{2}/"),
    re.compile(r"/\d{2}-\d{2}-20\d{2}/"),
    re.compile(r"/\d{1,2}-\d{1,2}-\d{4}/")
]

NEXT_LINK_INDICATORS = [
    "next", "next page", "more", "continue", "→", "›", ">",
    "page 2", "see more", "load more"
]


def is_event_link(href: str, text: str) -> bool:
    """
    Determine if a link is likely an event detail page.
    
    Args:
        href: The href attribute (URL)
        text: The anchor text
        
    Returns:
        True if link appears to be an event detail page
    """
    if not href:
        return False
    
    href_lower = href.lower()
    text_lower = text.lower().strip()
    
    # Check for event-related substrings in URL
    for substring in EVENT_LINK_SUBSTRINGS:
        if substring in href_lower:
            return True
    
    # Check for date patterns in URL
    for regex in DATE_SLUG_REGEXES:
        if regex.search(href):
            return True
    
    # Check for event-related keywords in link text
    for substring in EVENT_LINK_SUBSTRINGS:
        if substring in text_lower and len(text_lower) > 3:
            return True
    
    # Additional heuristics: look for date-like patterns in text
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}(/\d{4})?\b',  # 12/25 or 12/25/2024
        r'\b\d{1,2}-\d{1,2}(-\d{4})?\b',  # 12-25 or 12-25-2024
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b',  # Jan 25
        r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b'   # 25 Jan
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def is_next_link(href: str, text: str, rels: List[str]) -> bool:
    """
    Determine if a link is a pagination "next" link.
    
    Args:
        href: The href attribute (URL)
        text: The anchor text
        rels: List of rel attributes
        
    Returns:
        True if link appears to be a "next page" link
    """
    if not href:
        return False
    
    # Check rel attribute
    if "next" in rels:
        return True
    
    text_lower = text.lower().strip()
    
    # Check for common next page indicators
    for indicator in NEXT_LINK_INDICATORS:
        if indicator in text_lower:
            return True
    
    # Check for numeric pagination (page 2, page 3, etc.)
    if re.search(r'\bpage\s+\d+\b', text_lower) or re.search(r'\b\d+\b', text_lower):
        return True
    
    return False


def should_exclude_link(href: str, text: str) -> bool:
    """
    Determine if a link should be excluded from event discovery.
    
    Args:
        href: The href attribute (URL)
        text: The anchor text
        
    Returns:
        True if link should be excluded
    """
    if not href:
        return True
    
    href_lower = href.lower()
    text_lower = text.lower().strip()
    
    # Exclude common non-event links
    exclude_patterns = [
        "login", "register", "signup", "contact", "about", "privacy",
        "terms", "policy", "admin", "edit", "delete", "logout",
        "subscribe", "newsletter", "rss", "feed", "search",
        "category", "tag", "archive", "author", "user", "profile"
    ]
    
    for pattern in exclude_patterns:
        if pattern in href_lower or pattern in text_lower:
            return True
    
    # Exclude file downloads
    file_extensions = [".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png", ".gif"]
    if any(href_lower.endswith(ext) for ext in file_extensions):
        return True
    
    # Exclude very short or generic text
    if len(text_lower) <= 2 or text_lower in ["here", "link", "more", "read", "view"]:
        return True
    
    return False