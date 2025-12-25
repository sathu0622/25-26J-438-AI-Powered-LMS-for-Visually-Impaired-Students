# Document Processor Frontend

React frontend for the Document Processing application.

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure API URL

Create a `.env` file in the Frontend directory:

```
REACT_APP_API_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`

### 3. Run Development Server

```bash
npm start
```

The app will open at `http://localhost:3000`

### 4. Build for Production

```bash
npm run build
```

## Features

- Drag and drop file upload
- Real-time processing status
- Results display with:
  - Resource type detection with confidence score
  - Generated summaries
  - Extracted text
  - Article splitting for newspapers

## Supported File Types

- PDF
- JPG/JPEG
- PNG
- TIFF
- BMP

Maximum file size: 50MB





