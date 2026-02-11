# S3 Presigned URLs - Frontend Integration Guide

## Overview
The pipeline now generates **presigned S3 URLs** that expire after **30 minutes**. This allows the frontend to access files temporarily for testing without making the S3 bucket public.

## API Response Format

When you fetch a pipeline run status via `GET /pipeline/{run_id}`, the response now includes an `s3_data` field:

```json
{
  "run_id": "abc-123-xyz",
  "status": "COMPLETED",
  "s3_data": {
    "excel": {
      "url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz.xlsx",
      "presigned_url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz.xlsx?X-Amz-Algorithm=...",
      "expires_in": 1800
    },
    "json": {
      "url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz.json",
      "presigned_url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz.json?X-Amz-Algorithm=...",
      "expires_in": 1800
    },
    "debug_pdf": {
      "url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz_debug.pdf",
      "presigned_url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz_debug.pdf?X-Amz-Algorithm=...",
      "expires_in": 1800
    },
    "log_file": {
      "url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz.log",
      "presigned_url": "https://bucket.s3.eu-north-1.amazonaws.com/pipeline-outputs/abc-123-xyz/abc-123-xyz.log?X-Amz-Algorithm=...",
      "expires_in": 1800
    }
  }
}
```

## Frontend Usage

### Use Presigned URLs for Downloads
```javascript
// ✅ RECOMMENDED: Use presigned URLs (temporary, secure)
const downloadUrl = response.s3_data.excel.presigned_url;

// Download the file
window.open(downloadUrl, '_blank');
// or
fetch(downloadUrl)
  .then(res => res.blob())
  .then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'result.xlsx';
    a.click();
  });
```

### Permanent URLs (optional)
```javascript
// Permanent URL (requires bucket to be public or IAM permissions)
const permanentUrl = response.s3_data.excel.url;
```

## Important Notes

1. **Expiration**: Presigned URLs expire after **30 minutes (1800 seconds)**
2. **No Authentication Required**: Presigned URLs don't require AWS credentials
3. **Security**: URLs are signed and temporary, perfect for testing
4. **File Types**: Available for all pipeline outputs:
   - `excel`: Excel results file
   - `json`: JSON results file
   - `debug_pdf`: Debug PDF (if generated)
   - `log_file`: Processing logs (if generated)

## Checking URL Expiration

```javascript
const expiresIn = response.s3_data.excel.expires_in; // 1800 seconds
const expirationTime = new Date(Date.now() + expiresIn * 1000);
console.log(`URL expires at: ${expirationTime.toLocaleString()}`);
```

## Error Handling

If presigned URL generation fails, the field will be `null`:

```javascript
if (response.s3_data.excel.presigned_url === null) {
  console.warn('Presigned URL not available, using permanent URL');
  // Fallback to permanent URL or local endpoint
  downloadUrl = response.s3_data.excel.url || `/outputs/${filename}`;
}
```
