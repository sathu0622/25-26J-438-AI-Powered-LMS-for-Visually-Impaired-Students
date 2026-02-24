# Developer Guide: Adding New Features

This guide shows you how to extend the document module using the clean architecture pattern.

## Example: Adding "Export Document" Feature

Let's say we want users to export their processed document as PDF or DOCX.

### Step 1: Add API Service Method

**File:** `src/app/services/documentService.ts`

```tsx
// Add this interface
export interface ExportResponse {
  download_url: string;
  filename: string;
  format: 'pdf' | 'docx';
}

// Add this method to documentService
async exportDocument(
  documentId: string,
  format: 'pdf' | 'docx'
): Promise<ExportResponse> {
  return api.post<ExportResponse>('/export-document', {
    document_id: documentId,
    format,
  });
}
```

✅ **Benefit:** API endpoint is defined once, used from anywhere

### Step 2: Add State Handler in Hook

**File:** `src/app/hooks/useDocumentModule.ts`

```tsx
import { useCallback } from 'react';
import { documentService, ExportResponse } from '../services/documentService';

// Add this to the state interface
export interface DocumentModuleState {
  // ... existing state ...
  isExporting: boolean;
  exportError: string | null;
}

// In useDocumentModule function, add initial state
const [state, setState] = useState<DocumentModuleState>({
  // ... existing state ...
  isExporting: false,
  exportError: null,
});

// Add this handler
const handleExport = useCallback(async (format: 'pdf' | 'docx') => {
  if (!state.documentResult?.document_id) return;

  setState((prev) => ({
    ...prev,
    isExporting: true,
    exportError: null,
  }));

  try {
    const result = await documentService.exportDocument(
      state.documentResult.document_id,
      format
    );

    // Trigger download
    const link = document.createElement('a');
    link.href = result.download_url;
    link.download = result.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Show success message
    safeSpeak(`Document exported as ${format.toUpperCase()}`);
  } catch (err) {
    const message =
      err instanceof Error
        ? err.message
        : `Failed to export document. Please try again.`;

    setState((prev) => ({
      ...prev,
      exportError: message,
    }));
  } finally {
    setState((prev) => ({
      ...prev,
      isExporting: false,
    }));
  }
}, [state.documentResult]);

// Return the handler from the hook
return {
  // ... existing returns ...
  handleExport,
  isExporting: state.isExporting,
  exportError: state.exportError,
};
```

✅ **Benefit:** All logic in one place, easy to test

### Step 3: Create Presentation Component

**File:** `src/app/components/document/DocumentExport.tsx`

```tsx
import { Download, Loader2 } from 'lucide-react';
import { Button } from '../ui/button';

interface DocumentExportProps {
  onExport: (format: 'pdf' | 'docx') => void;
  isLoading: boolean;
}

export const DocumentExport = ({
  onExport,
  isLoading,
}: DocumentExportProps) => {
  return (
    <div className="space-y-3">
      <h2 className="text-center">Export Document</h2>
      <div className="grid gap-3 sm:grid-cols-2">
        <Button
          onClick={() => onExport('pdf')}
          disabled={isLoading}
          size="lg"
          className="min-h-[72px] flex-col gap-2"
          aria-label="Export document as PDF"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-6 w-6 animate-spin" aria-hidden="true" />
              <span>Exporting...</span>
            </>
          ) : (
            <>
              <Download className="h-6 w-6" aria-hidden="true" />
              <span>Export as PDF</span>
            </>
          )}
        </Button>
        <Button
          onClick={() => onExport('docx')}
          disabled={isLoading}
          size="lg"
          variant="outline"
          className="min-h-[72px] flex-col gap-2"
          aria-label="Export document as Word"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-6 w-6 animate-spin" aria-hidden="true" />
              <span>Exporting...</span>
            </>
          ) : (
            <>
              <Download className="h-6 w-6" aria-hidden="true" />
              <span>Export as DOCX</span>
            </>
          )}
        </Button>
      </div>
    </div>
  );
};
```

✅ **Benefit:** Pure presentation, no logic, fully reusable

### Step 4: Add to Container

**File:** `src/app/components/document/DocumentModule.tsx`

```tsx
import { DocumentExport } from './DocumentExport';

export const DocumentModule = () => {
  const {
    // ... existing destructuring ...
    handleExport,
    isExporting,
    exportError,
  } = useDocumentModule();

  return (
    <>
      {/* ... existing error handling ... */}
      {exportError && (
        <Card className="border-destructive bg-destructive/10 p-4">
          <p className="font-medium">Export error</p>
          <p className="text-sm">{exportError}</p>
        </Card>
      )}

      {/* ... existing screens ... */}

      {/* Add this after DocumentQA screen */}
      {screen === 'summary' && (
        <>
          <DocumentSummary
            summary={documentSummary}
            onAskQuestion={handleStartQA}
            articles={documentResult?.article_list}
            selectedArticleId={selectedArticleId}
            onSelectArticle={handleSelectArticle}
          />
          <DocumentExport
            onExport={handleExport}
            isLoading={isExporting}
          />
        </>
      )}
    </>
  );
};
```

✅ **Benefit:** Easy to add new components to the workflow

### Done! 🎉

You've just added a new feature following the clean architecture pattern!

## Pattern Summary

The pattern is always:
1. **Service Method** - Define API call
2. **Hook Handler** - Manage state for that feature
3. **Component** - Create pure UI for that feature
4. **Container** - Add component to workflow

This ensures:
- Testable code
- No repeated logic
- Easy to maintain
- Professional organization

## Testing This Feature

### Unit Test - Service
```tsx
// documentService.test.ts
test('exportDocument should call correct endpoint', async () => {
  const result = await documentService.exportDocument('doc-123', 'pdf');
  expect(result).toHaveProperty('download_url');
  expect(result.format).toBe('pdf');
});
```

### Unit Test - Hook
```tsx
// useDocumentModule.test.ts
test('handleExport should trigger download', async () => {
  const { result } = renderHook(() => useDocumentModule());
  
  act(() => {
    result.current.handleExport('pdf');
  });
  
  await waitFor(() => {
    expect(result.current.isExporting).toBe(false);
  });
});
```

### Component Test - Presentation
```tsx
// DocumentExport.test.tsx
test('DocumentExport should call onExport when button clicked', () => {
  const mockOnExport = jest.fn();
  render(<DocumentExport onExport={mockOnExport} isLoading={false} />);
  
  userEvent.click(screen.getByLabelText('Export document as PDF'));
  
  expect(mockOnExport).toHaveBeenCalledWith('pdf');
});
```

✅ **Benefit:** Each layer is independently testable

## Common Patterns

### Pattern 1: Data Fetching
```tsx
// service.ts
async fetchData(id) { ... }

// hook.ts
const handleFetch = async () => {
  setState(prev => ({ ...prev, isLoading: true }));
  try {
    const data = await service.fetchData(id);
    setState(prev => ({ ...prev, data, error: null }));
  } catch (err) {
    setState(prev => ({ ...prev, error: err.message }));
  }
}

// component.tsx
<Component data={data} onFetch={handleFetch} />
```

### Pattern 2: Form Submission
```tsx
// service.ts
async submitForm(data) { ... }

// hook.ts
const handleSubmit = async (formData) => {
  setState(prev => ({ ...prev, isSubmitting: true }));
  try {
    const result = await service.submitForm(formData);
    setState(prev => ({ ...prev, result, error: null }));
  } catch (err) {
    setState(prev => ({ ...prev, error: err.message }));
  }
}

// component.tsx
<Form onSubmit={handleSubmit} isLoading={isSubmitting} />
```

### Pattern 3: Multiple Operations
```tsx
// service.ts
async operation1() { ... }
async operation2() { ... }

// hook.ts
const handleOperationFlow = async () => {
  setState(prev => ({ ...prev, step: 1 }));
  const data1 = await service.operation1();
  const data2 = await service.operation2(data1);
  setState(prev => ({ ...prev, step: 2, result: data2 }));
}

// container.tsx
{step === 1 && <Component1 />}
{step === 2 && <Component2 />}
```

## When to Create New Files

Create a new service method when:
- ✓ You need to call a new API endpoint
- ✓ The logic is used in multiple places
- ✓ The logic is unrelated to current functionality

Create a new hook when:
- ✓ You have complex state logic
- ✓ Multiple components need the same state
- ✓ The logic is reusable across screens

Create a new component when:
- ✓ You have new UI to show
- ✓ The UI is reusable
- ✓ The component is independent

## Checklist for New Features

- [ ] Created API method in `documentService.ts`
- [ ] Added handler in `useDocumentModule.ts`
- [ ] Created presentation component
- [ ] Added component to `DocumentModule.tsx`
- [ ] Updated TypeScript interfaces
- [ ] Added error handling
- [ ] Added loading states
- [ ] Added accessibility labels (aria-label, aria-live, etc.)
- [ ] Tested in browser
- [ ] Followed naming conventions
- [ ] Added JSDoc comments if needed

Following this pattern ensures professional, maintainable, and testable code! 🚀
