"""
Event tags configuration for consistent categorization.
"""

EVENT_TYPES = [
    "Class",  # includes workshop training, courses, tutorials
    "Conference",
    "Talk",  # includes panel, lectures, discussions
    "Show",  # includes performance, concerts, theater, dance, comedy, screening
    "Exhibition",  # includes art shows, installations, galleries, museum exhibits
    "Party",  # includes social, mixers, receptions, gatherings
    "Networking",  # professional meetups
    "Festival",  # includes fair, markets, expos
    "Tour",  # includes walking tours, tastings, open houses
    "Competition",  # includes contests, tournaments, hackathons
]

TOPICS = [
    "Visual Arts",  # painting, sculpture, photography, mixed media, ceramics, textiles
    "Fashion",  # includes jewelry, fashion
    "Performing Arts",  # theater, dance, circus, cabaret
    "Music",  # all genres, concerts, DJ sets
    "Comedy",  # standup, improv, sketch
    "Film",  # cinema, video art, digital media
    "Literature",  # readings, book launches, poetry
    "Food & Drink",  # culinary events, tastings, dining
    "Education",  # learning, workshops, professional development
    "Business & Tech",  # creative industries, arts business, digital arts
    "Community",  # local culture, activism, social causes
    "Family",  # kids programming, all-ages events
]


def validate_event_type(event_type: str) -> bool:
    """
    Validate that an event type is in the allowed list.
    
    Args:
        event_type: The event type to validate
        
    Returns:
        True if valid, False otherwise
    """
    return event_type in EVENT_TYPES


def validate_topics(topics: list) -> bool:
    """
    Validate that all topics are in the allowed list.
    
    Args:
        topics: List of topics to validate
        
    Returns:
        True if all topics are valid, False otherwise
    """
    if not isinstance(topics, list):
        return False
    
    return all(topic in TOPICS for topic in topics)


def get_event_types() -> list:
    """Get all available event types."""
    return EVENT_TYPES.copy()


def get_topics() -> list:
    """Get all available topics."""
    return TOPICS.copy()


def format_tags_for_prompt() -> str:
    """
    Format tags for inclusion in LLM prompts.
    
    Returns:
        Formatted string with event types and topics
    """
    event_types_str = ", ".join(f'"{t}"' for t in EVENT_TYPES)
    topics_str = ", ".join(f'"{t}"' for t in TOPICS)
    
    return f"""Available Event Types: {event_types_str}

Available Topics: {topics_str}"""