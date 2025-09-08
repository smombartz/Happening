// Global variables
let allEvents = [];
let fuse;
let currentEvents = [];

// Handlebars template
const eventTemplate = `
<div class="events-grid">
  {{#each events}}
  <div class="event-card">
    {{#if extraction.image_url}}
    <img src="{{extraction.image_url}}" alt="{{extraction.title}}" class="event-image" loading="lazy">
    {{else}}
    <div class="event-image"></div>
    {{/if}}
    <div class="event-content">
      <div class="event-title">{{extraction.title}}</div>
      {{#if extraction.start_date}}
      <div class="event-date">
        {{formatDate extraction.start_date extraction.end_date extraction.start_time}}
      </div>
      {{/if}}
      {{#if extraction.location_address}}
      <div class="event-location">{{extraction.location_address}}</div>
      {{/if}}
      {{#if extraction.description}}
      <div class="event-description">{{truncateDescription extraction.description}}</div>
      {{/if}}
      {{#if extraction.detail_url}}
      <a href="{{extraction.detail_url}}" target="_blank" class="event-link">View Details →</a>
      {{/if}}
    </div>
  </div>
  {{/each}}
</div>
`;

// Handlebars helpers
Handlebars.registerHelper('formatDate', function(startDate, endDate, startTime) {
    if (!startDate) return '';
    
    const start = new Date(startDate);
    const options = { 
        weekday: 'short', 
        month: 'short', 
        day: 'numeric',
        year: start.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
    };
    
    let formatted = start.toLocaleDateString('en-US', options);
    
    if (endDate && endDate !== startDate) {
        const end = new Date(endDate);
        if (start.getMonth() === end.getMonth() && start.getFullYear() === end.getFullYear()) {
            formatted += ` - ${end.getDate()}`;
        } else {
            formatted += ` - ${end.toLocaleDateString('en-US', options)}`;
        }
    }
    
    if (startTime) {
        const time = new Date(`2000-01-01T${startTime}`);
        formatted += ` at ${time.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}`;
    }
    
    return formatted;
});

Handlebars.registerHelper('truncateDescription', function(description) {
    if (!description) return '';
    return description.length > 120 ? description.substring(0, 120) + '...' : description;
});

// Load and parse events from JSONL file
async function loadEvents() {
    try {
        const response = await fetch('events.jsonl');
        const text = await response.text();
        
        // Parse JSONL - each line is a JSON object
        const lines = text.trim().split('\n').filter(line => line.trim());
        allEvents = lines.map(line => JSON.parse(line));
        
        // Filter events that have extraction data
        allEvents = allEvents.filter(event => event.extraction && event.extraction.title);
        
        // Sort events by date
        allEvents.sort((a, b) => {
            const dateA = new Date(a.extraction.start_date || '1900-01-01');
            const dateB = new Date(b.extraction.start_date || '1900-01-01');
            return dateA - dateB;
        });
        
        currentEvents = [...allEvents];
        
        // Initialize Fuse.js for search
        initializeSearch();
        
        // Render initial events
        renderEvents();
        
    } catch (error) {
        console.error('Error loading events:', error);
        document.getElementById('eventsContainer').innerHTML = `
            <div class="no-results">
                <p>Error loading events. Please make sure events.jsonl is available.</p>
            </div>
        `;
    }
}

// Initialize Fuse.js search
function initializeSearch() {
    const options = {
        keys: [
            { name: 'extraction.title', weight: 0.4 },
            { name: 'extraction.description', weight: 0.3 },
            { name: 'extraction.location_address', weight: 0.2 },
            { name: 'content_markdown', weight: 0.1 }
        ],
        threshold: 0.4,
        includeScore: true,
        minMatchCharLength: 2
    };
    
    fuse = new Fuse(allEvents, options);
}

// Render events using Handlebars template
function renderEvents() {
    const container = document.getElementById('eventsContainer');
    
    if (currentEvents.length === 0) {
        container.innerHTML = `
            <div class="no-results">
                <p>No events found. Try adjusting your search.</p>
            </div>
        `;
        return;
    }
    
    const template = Handlebars.compile(eventTemplate);
    const html = template({ events: currentEvents });
    container.innerHTML = html;
}

// Handle search input
function handleSearch(query) {
    if (!query || query.trim() === '') {
        currentEvents = [...allEvents];
    } else {
        const results = fuse.search(query.trim());
        currentEvents = results.map(result => result.item);
    }
    
    renderEvents();
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    
    // Debounced search
    let searchTimeout;
    searchInput.addEventListener('input', function(e) {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            handleSearch(e.target.value);
        }, 300);
    });
    
    // Load events on page load
    loadEvents();
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Focus search on '/' key
    if (e.key === '/' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        document.getElementById('searchInput').focus();
    }
    
    // Clear search on Escape
    if (e.key === 'Escape' && e.target.id === 'searchInput') {
        e.target.value = '';
        handleSearch('');
    }
});