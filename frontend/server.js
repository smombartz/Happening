const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3000;

// MIME types mapping
const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.jsonl': 'application/jsonl',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

const server = http.createServer((req, res) => {
    // Parse URL and remove query string
    let filePath = req.url.split('?')[0];
    
    // Default to index.html for root path
    if (filePath === '/') {
        filePath = '/index.html';
    }
    
    // Construct full file path
    const fullPath = path.join(__dirname, filePath);
    
    // Get file extension
    const extname = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[extname] || 'application/octet-stream';
    
    // Check if file exists
    fs.access(fullPath, fs.constants.F_OK, (err) => {
        if (err) {
            // File not found
            res.writeHead(404, { 'Content-Type': 'text/html' });
            res.end('<h1>404 - File Not Found</h1>');
            return;
        }
        
        // Read and serve the file
        fs.readFile(fullPath, (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'text/html' });
                res.end('<h1>500 - Internal Server Error</h1>');
                return;
            }
            
            // Set CORS headers to allow cross-origin requests
            res.writeHead(200, {
                'Content-Type': contentType,
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            });
            
            res.end(data);
        });
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
    console.log('Press Ctrl+C to stop the server');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down server...');
    server.close(() => {
        console.log('Server stopped.');
        process.exit(0);
    });
});