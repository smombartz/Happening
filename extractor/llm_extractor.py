"""
LLM-based field extraction using local Ollama Mistral model.
"""
import json
import re
import time
from typing import Dict, Any, Optional, List
import httpx
from pydantic import BaseModel, Field, ValidationError, validator
from datetime import datetime
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

from utils.io_utils import log_info, log_warn, log_error, log_debug, print_substep, print_spinner_message, print_spinner_complete
from utils.url_utils import clean_url_for_display
from config.tags import EVENT_TYPES, TOPICS, validate_event_type, validate_topics, format_tags_for_prompt


class EventExtraction(BaseModel):
    """Pydantic model for event extraction validation."""
    title: Optional[str] = Field(None, description="Event title")
    start_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$|^null$', description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$|^null$', description="End date (YYYY-MM-DD)")
    start_time: Optional[str] = Field(None, pattern=r'^\d{2}:\d{2}$|^null$', description="Start time (HH:MM)")
    end_time: Optional[str] = Field(None, pattern=r'^\d{2}:\d{2}$|^null$', description="End time (HH:MM)")
    timezone: Optional[str] = Field(None, description="Timezone (IANA format)")
    location_address: Optional[str] = Field(None, description="Location address")
    image_url: Optional[str] = Field(None, description="Event image URL")
    description: Optional[str] = Field(None, max_length=500, description="Event description (max 500 chars)")
    event_type: Optional[str] = Field(None, description="Event type from predefined list")
    topics: Optional[List[str]] = Field(default_factory=list, description="Event topics from predefined list")
    source_url: str = Field(..., description="Source listing URL")
    detail_url: str = Field(..., description="Event detail URL")
    
    @validator('event_type')
    def validate_event_type_field(cls, v):
        if v is not None and v not in EVENT_TYPES:
            return None  # Return None for invalid types rather than raising error
        return v
    
    @validator('topics')
    def validate_topics_field(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        # Filter out invalid topics
        return [topic for topic in v if topic in TOPICS]


class LLMExtractor:
    """Extractor using local Ollama Mistral model for structured event field extraction."""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434", 
                 model_name: str = "mistral",
                 timeout: float = 120.0):
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.session = httpx.Client(timeout=timeout)
    
    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()
    
    def _build_extraction_prompt(self, markdown: str, source_url: str, detail_url: str) -> str:
        """
        Build the extraction prompt for the LLM.
        
        Args:
            markdown: Page content as markdown
            source_url: Source listing URL
            detail_url: Event detail URL
            
        Returns:
            Formatted prompt string
        """
        # Truncate markdown if too long (to avoid token limits)
        max_markdown_length = 8000
        if len(markdown) > max_markdown_length:
            markdown = markdown[:max_markdown_length] + "\n\n[...content truncated...]"
        
        tags_info = format_tags_for_prompt()
        
        prompt = f"""You are given:
- detail_url: {detail_url}
- source_url: {source_url}
- page_markdown:
\"\"\"
{markdown}
\"\"\"

{tags_info}

Extract and return a single JSON object with exactly these keys:
{{
    "title": string|null,
    "start_date": string|null,
    "end_date": string|null,
    "start_time": string|null,
    "end_time": string|null,
    "timezone": string|null,
    "location_address": string|null,
    "image_url": string|null,
    "description": string|null,
    "event_type": string|null,
    "topics": array of strings|null,
    "source_url": string,
    "detail_url": string
}}

Rules:
- CRITICAL: Dates must be in YYYY-MM-DD format only. Examples: "2025-08-08", "2025-12-31". Never use formats like "August 8, 2025" or "Aug 8, 2025".
- If date is a range like "June 2–4, 2025", convert to ISO format: start_date="2025-06-02", end_date="2025-06-04".
- If only a single date is present, set end_date = start_date (both in YYYY-MM-DD format).
- If only a month/day without year is given, infer year from page context (URL or metadata) if possible; otherwise null.
- Times like "7pm" -> "19:00". If an end time isn't present but duration is (e.g., "2 hours"), compute end_time when start_time is known.
- Address: include venue name and city/state if present; otherwise the best available location string.
- Prefer og:image or twitter:image for image_url; else first main content image.
- Keep description <= 500 chars, plain text, no markdown.
- If multiple events are present, choose the primary event (largest heading or first dated block).
- For event_type: choose exactly ONE type from the Available Event Types list that best matches this event.
- For topics: choose 1-3 topics from the Available Topics list that best describe this event. Return as an array.
- IMPORTANT: Only use event types and topics from the provided lists above. Do not create new categories.

Return only the JSON object, no additional text."""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call local Ollama API with the extraction prompt.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Raw response text from the model
            
        Raises:
            httpx.HTTPError: If API call fails
        """
        system_message = """You are a precise event data extractor. You read Markdown content from a single event page and return a strict JSON object with normalized fields. If a field is unknown, use null. Dates must be ISO 8601 (YYYY-MM-DD). Times must be 24-hour format HH:MM. Never include markdown or prose outside the JSON."""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_message,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 1000
            }
        }
        
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            if 'response' not in result:
                raise ValueError(f"Unexpected Ollama response format: {result}")
            
            return result['response'].strip()
            
        except httpx.HTTPError as e:
            log_error(f"Ollama API call failed: {e}")
            raise
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse Ollama response JSON: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract and validate JSON from LLM response.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed and validated JSON dictionary
            
        Raises:
            ValueError: If JSON is invalid or missing
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}...")
        
        json_str = json_match.group(0)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log_debug(f"Raw JSON that failed to parse: {json_str}")
            raise ValueError(f"Invalid JSON in response: {e}")
    
    def _validate_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data using Pydantic model.
        
        Args:
            data: Raw extracted data dictionary
            
        Returns:
            Validated data dictionary
            
        Raises:
            ValidationError: If data doesn't match schema
        """
        try:
            # Convert None strings to actual None
            cleaned_data = {}
            for key, value in data.items():
                if value == "null" or value == "" or value is None:
                    cleaned_data[key] = None
                else:
                    cleaned_data[key] = value
            
            extraction = EventExtraction(**cleaned_data)
            return extraction.model_dump()
            
        except ValidationError as e:
            log_error(f"Validation failed: {e}")
            raise
    
    def _post_process_extraction(self, data: Dict[str, Any], image_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Post-process the extracted data for consistency.
        
        Args:
            data: Validated extraction data
            image_hint: Fallback image URL if extraction didn't find one
            
        Returns:
            Post-processed data
        """
        processed = data.copy()
        
        # Use image hint if no image_url was extracted
        if not processed.get('image_url') and image_hint:
            processed['image_url'] = image_hint
            log_debug(f"Using image hint: {image_hint}")
        
        # Ensure end_date equals start_date if not provided but start_date exists
        if processed.get('start_date') and not processed.get('end_date'):
            processed['end_date'] = processed['start_date']
            log_debug("Set end_date = start_date for single-day event")
        
        # Clean up description
        if processed.get('description'):
            desc = processed['description'].strip()
            # Remove markdown-like formatting
            desc = re.sub(r'[*_`#\[\]()]', '', desc)
            # Limit length
            if len(desc) > 500:
                desc = desc[:497] + "..."
            processed['description'] = desc if desc else None
        
        # Validate and clean tags (additional safety check)
        if processed.get('event_type') and processed['event_type'] not in EVENT_TYPES:
            log_debug(f"Invalid event_type '{processed['event_type']}' replaced with None")
            processed['event_type'] = None
        
        if processed.get('topics'):
            if not isinstance(processed['topics'], list):
                processed['topics'] = []
            else:
                # Filter out invalid topics
                valid_topics = [topic for topic in processed['topics'] if topic in TOPICS]
                if len(valid_topics) != len(processed['topics']):
                    invalid_topics = [topic for topic in processed['topics'] if topic not in TOPICS]
                    log_debug(f"Removed invalid topics: {invalid_topics}")
                processed['topics'] = valid_topics
        else:
            processed['topics'] = []
        
        # Parse and normalize dates if needed
        for date_field in ['start_date', 'end_date']:
            date_value = processed.get(date_field)
            if date_value and not self._is_iso_date_format(date_value):
                parsed_date = self._parse_date_to_iso(date_value)
                if parsed_date:
                    log_debug(f"Converted {date_field} from '{date_value}' to '{parsed_date}'")
                    processed[date_field] = parsed_date
                else:
                    log_debug(f"Failed to parse {date_field} '{date_value}', setting to None")
                    processed[date_field] = None
        
        return processed
    
    def _is_iso_date_format(self, date_str: str) -> bool:
        """Check if date string is in YYYY-MM-DD format."""
        if not date_str or date_str == "null":
            return True
        return re.match(r'^\d{4}-\d{2}-\d{2}$', date_str) is not None
    
    def _parse_date_to_iso(self, date_str: str) -> Optional[str]:
        """
        Parse various date formats to ISO YYYY-MM-DD format.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            ISO formatted date string or None if parsing fails
        """
        if not date_str or date_str.lower() == "null":
            return None
        
        # Try dateutil parser if available
        if HAS_DATEUTIL:
            try:
                parsed = date_parser.parse(date_str)
                return parsed.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                pass
        
        # Fallback: try common formats manually
        common_formats = [
            '%B %d, %Y',     # "August 8, 2025"
            '%b %d, %Y',     # "Aug 8, 2025"  
            '%m/%d/%Y',      # "8/8/2025"
            '%m-%d-%Y',      # "8-8-2025"
            '%Y/%m/%d',      # "2025/8/8"
        ]
        
        for fmt in common_formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        log_debug(f"Could not parse date: {date_str}")
        return None
    
    def extract(self, markdown: str, source_url: str, detail_url: str, 
                image_hint: Optional[str] = None, max_retries: int = 1) -> Dict[str, Any]:
        """
        Extract structured event fields from markdown content.
        
        Args:
            markdown: Event page content as markdown
            source_url: Source listing URL
            detail_url: Event detail URL
            image_hint: Optional fallback image URL
            max_retries: Maximum number of retries if JSON is invalid
            
        Returns:
            Dictionary with extracted event fields
        """
        log_debug(f"Extracting fields from {clean_url_for_display(detail_url)}")
        
        prompt = self._build_extraction_prompt(markdown, source_url, detail_url)
        log_debug(f"Built prompt: {len(prompt)} chars")
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    log_debug(f"Retry {attempt + 1}/{max_retries + 1} for extraction: {clean_url_for_display(detail_url)}")
                
                # Call LLM with timing
                llm_start = time.time()
                log_debug(f"Sending {len(markdown)} chars of content to LLM...")
                response = self._call_ollama(prompt)
                llm_time = time.time() - llm_start
                
                log_debug(f"LLM responded in {llm_time:.1f}s, response length: {len(response)} chars")
                
                # Extract and validate JSON
                log_debug(f"Parsing JSON response...")
                raw_data = self._extract_json_from_response(response)
                log_debug(f"Validating extracted data...")
                validated_data = self._validate_extraction(raw_data)
                
                # Post-process
                final_data = self._post_process_extraction(validated_data, image_hint)
                
                # Log successful extraction details
                title = final_data.get('title', 'No title')
                date = final_data.get('start_date', 'No date')
                location = final_data.get('location_address', 'No location')
                
                log_debug(f"Successfully extracted: '{title}' on {date} at {location}")
                return final_data
                
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                log_debug(f"Extraction attempt {attempt + 1} failed for {clean_url_for_display(detail_url)}: {e}")
                
                if attempt == max_retries:
                    # Return minimal valid data on final failure
                    log_debug(f"All extraction attempts failed for {clean_url_for_display(detail_url)}")
                    return {
                        "title": None,
                        "start_date": None,
                        "end_date": None,
                        "start_time": None,
                        "end_time": None,
                        "timezone": None,
                        "location_address": None,
                        "image_url": image_hint,
                        "description": None,
                        "event_type": None,
                        "topics": [],
                        "source_url": source_url,
                        "detail_url": detail_url
                    }
            
            except Exception as e:
                log_debug(f"Unexpected error in extraction for {clean_url_for_display(detail_url)}: {e}")
                
                if attempt == max_retries:
                    return {
                        "title": None,
                        "start_date": None,
                        "end_date": None,
                        "start_time": None,
                        "end_time": None,
                        "timezone": None,
                        "location_address": None,
                        "image_url": image_hint,
                        "description": None,
                        "event_type": None,
                        "topics": [],
                        "source_url": source_url,
                        "detail_url": detail_url
                    }


def extract(markdown: str, source_url: str, detail_url: str, 
           image_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract event fields from markdown.
    
    Args:
        markdown: Event page content as markdown
        source_url: Source listing URL  
        detail_url: Event detail URL
        image_hint: Optional fallback image URL
        
    Returns:
        Dictionary with extracted event fields
    """
    extractor = LLMExtractor()
    return extractor.extract(markdown, source_url, detail_url, image_hint)