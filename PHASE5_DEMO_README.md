# Phase 5 Demo - Complete Area Calculation

**Standalone Phase 5 demonstration with all configurations hardcoded**

You needs to: **Upload PDF** + **Enter Page Number** = **Get Full Calculations**

---

## 🚀 Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run the demo (on port 7861)
python phase5_demo.py
```

Open: **http://localhost:7861**

---

## 📖 How to Use

1. **Upload PDF** - Any architectural drawing PDF
2. **Enter Page Number** - Which page to analyze (1, 2, 3, etc.)
3. **Click "Run Phase 5 Calculation"**
4. **Get Results:**
   - Line items table with square footage breakdown
   - Grand total net area
   - Visual debug with green overlays
   - All PDF pages for reference

---

## 🎯 What's Included (Hardcoded)

### Scale Data
```
48.5 pixels = 1 foot (for all pages)
```

### Window Specifications
- **W1**: 3' × 5' (15 sqft each)
- **W2**: 4' × 6' (24 sqft each)
- **W3**: 5' × 7' (35 sqft each)

### Door Specifications
- **A**: 3' × 7' (21 sqft each)
- **B**: 6' × 8' (48 sqft - overhead)
- **D1**: 3.5' × 8' (28 sqft each)
- **D2**: 4' × 8' (32 sqft each)

### Survey Data (Tags per Page)
Hardcoded tag counts for pages 1-10. Example:
- Page 1: 12 "A" doors, 8 "B" doors, 15 "W1" windows, etc.

---

## 📊 Example Output

**Input:**
- PDF: `MyProject.pdf`
- Page: `1`

**Output:**
```
Grand Total Net Area: 4,235.50 sqft

Line Items:
1. Gross Facade Area (Vector) - 5,120.00 sqft
2. Window W1 (Deduction) - 15 × 15sf = -225.00 sqft
3. Window W2 (Deduction) - 10 × 24sf = -240.00 sqft
4. Door A (Deduction) - 12 × 21sf = -252.00 sqft
... (etc)
```

Plus visual debug showing SAM3 detection with green overlay!

---

## 🔍 What Phase 5 Does

1. **SAM3 Detection** - Automatically finds building regions on the page
2. **Gross Calculation** - Measures total building area using SAM3 mask
3. **Deduction Calculation** - Subtracts windows/doors based on hardcoded specs
4. **Smart Clamping** - Limits deductions to max 40% of gross area
5. **Line Item Generation** - Creates detailed takeoff breakdown

---

## 🖼️ Visual Output

**Gallery shows:**
1. All PDF pages (with status: Processed or Preview)
2. SAM3 results with GREEN OVERLAY showing detected building area
3. Captions with net square footage

---

## ⚙️ SAM3 Configuration

**Status panel shows:**
- ✅ PyTorch installed
- ✅ CUDA available
- ✅ Model files found
- ✅ SAM3 loaded

**If SAM3 not configured:**
- Will automatically fall back to VLM (Ollama)
- Then to full-page detection if VLM unavailable
- Pipeline always works regardless

---

## 📋 For Your Manager

**Perfect demonstration because:**

✅ **Zero Setup** - All inputs hardcoded, no JSON needed
✅ **Simple Interface** - Just PDF + page number
✅ **Complete Workflow** - Shows entire Phase 5 calculation
✅ **Visual Proof** - Green overlays show exactly what was measured
✅ **Professional Output** - Clean tables and metrics
✅ **Self-Diagnostic** - Shows SAM3 status automatically

**Tell them:**
- "Upload your architectural PDF"
- "Type a page number (like 1 or 10)"
- "Click the button"
- "See complete area calculations with SAM3!"

---

---

## 🐛 Troubleshooting

**"No line items generated"**
- Check page number is valid
- Try a different page with clearer drawings

**"SAM3 not available"**
- Will use VLM fallback automatically
- Results still work, just different detection method

**"Net area is 0"**
- Page may not have detected building regions
- Try adjusting hardcoded scale or survey data in the file

---

## 📦 Dependencies

**Required:**
- `pipeline/phase5_v2.py` (already exists)
- `pipeline/sam3_segmentation.py` (already exists)
- All other pipeline files

**This file is standalone** - it just calls Phase 5 with hardcoded inputs!
